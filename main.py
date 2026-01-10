#!/usr/bin/env python3
"""
CourtListener Standalone Test App (v2.0 - Fixed Verification Logic)
====================================================================

Isolated test environment for debugging CourtListener API integration.
Deploy on Railway to test independently of QuotationGenie.

Endpoints:
    GET  /           - Web UI for testing
    POST /search     - Search by quote text
    POST /citation   - Lookup by citation (e.g., "388 U.S. 1")
    GET  /health     - Health check
    GET  /config     - Show configuration status

Version History:
    2026-01-06 V2.0: FIXED verification logic - find_quote_in_source()
                     Now locates USER's quote in full text (not snippet position)
                     Eliminates false "deletion" diffs from buffer regions
                     Buffer reduced from ±200 to ±50 chars (edge comparison only)
    2026-01-04 V1.9: Side-by-side comparison UI (user quote vs authentic source)
    2026-01-04 V1.8: Full text verification - fetches opinion, extracts ±200 char buffer, catches all errors
    2026-01-04 V1.7.1: Bug fix - always compute diffs even for exact phrase matches
    2026-01-04 V1.7: Trace display, returns best matches with error highlighting (like Google Books)
    2026-01-04 V1.6: Error detection with diff highlighting (aligned with Google Books)
    2026-01-04 V1.5: Em-dash splitting (fixes "basin—Seymour" blocking "buspirone")
    2026-01-04 V1.4: Dynamic threshold adjusts for length mismatch
    2026-01-04 V1.3: Distinctive word anchoring
    2026-01-04 V1.2: Unicode + Fuzzy Matching
    2026-01-04 V1.1: Strip quotation marks from user input
"""

import os
import re
import html
import logging
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

# NER disabled - using regex-based extraction instead
NER_AVAILABLE = False
nlp = None

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Check both possible API key names
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "")
CL_API_KEY = os.getenv("CL_API_KEY", "")

# Use whichever is set (prefer COURTLISTENER_API_KEY)
if COURTLISTENER_API_KEY:
    API_KEY_SOURCE = "COURTLISTENER_API_KEY"
    ACTIVE_API_KEY = COURTLISTENER_API_KEY
elif CL_API_KEY:
    API_KEY_SOURCE = "CL_API_KEY"
    ACTIVE_API_KEY = CL_API_KEY
else:
    API_KEY_SOURCE = None
    ACTIVE_API_KEY = ""

COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
API_TIMEOUT = 30.0

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DiffSegment:
    """Represents a difference between user quote and source."""
    position: int           # Character position in user's quote
    user_text: str          # What user wrote
    source_text: str        # What source says
    diff_type: str          # 'substitution', 'insertion', 'deletion'

@dataclass
class SearchResult:
    """Result from CourtListener search."""
    success: bool
    case_name: str
    citation: str
    court: str
    date_filed: str
    snippet: str
    url: str
    cluster_id: str
    match_score: float = 0.0  # 0.0 - 1.0 similarity score
    error: str = ""
    diffs: List[Dict[str, Any]] = field(default_factory=list)  # Diff details as dicts
    verified_quote: str = ""  # User's quote with diffs marked
    source_quote: str = ""  # Authentic text from source for side-by-side

@dataclass
class SearchResponse:
    """Response containing results and trace log."""
    results: List[SearchResult]
    trace: List[str] = field(default_factory=list)

# =============================================================================
# COURT MAPPINGS
# =============================================================================

COURT_NAME_MAP = {
    'scotus': 'Supreme Court of the United States',
    'ca1': 'First Circuit',
    'ca2': 'Second Circuit',
    'ca3': 'Third Circuit',
    'ca4': 'Fourth Circuit',
    'ca5': 'Fifth Circuit',
    'ca6': 'Sixth Circuit',
    'ca7': 'Seventh Circuit',
    'ca8': 'Eighth Circuit',
    'ca9': 'Ninth Circuit',
    'ca10': 'Tenth Circuit',
    'ca11': 'Eleventh Circuit',
    'cadc': 'D.C. Circuit',
    'cafc': 'Federal Circuit',
}

# =============================================================================
# TEXT CLEANING AND MATCHING (aligned with google_books.py)
# =============================================================================

MATCH_THRESHOLD = 0.90  # 90% minimum for fuzzy matching

# Pattern for dashes that should split text (like ellipsis does)
# Em-dash often joins clauses without spaces: "basin—Seymour"
# Also catches hyphen since browsers/forms often convert em-dash to hyphen
DASH_SPLIT_PATTERN = re.compile(r'[-—–―]')  # Hyphen, em-dash, en-dash, horizontal bar


def clean_quote_text(text: str) -> str:
    """Clean special characters for API acceptance while preserving Unicode symbols."""
    text = text.strip().strip('"').strip('\u201C').strip('\u201D')
    text = text.replace('\u2019', "'").replace('\u2018', "'")  # Curly apostrophes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Curly quotes
    # NOTE: We no longer convert em-dash to hyphen here - we split at it instead
    # Remove double quotes (we wrap in our own for exact phrase search)
    text = text.replace('"', '')
    # Normalize Unicode to NFC (preserves §, ¶, accented chars)
    text = unicodedata.normalize('NFC', text)
    return ' '.join(text.split())  # Normalize whitespace


def compute_match_score(user_quote: str, source_text: str) -> float:
    """Compute similarity score between user quote and source text."""
    if not user_quote or not source_text:
        return 0.0
    
    # Normalize for comparison
    def normalize(t):
        t = t.lower().strip()
        t = t.replace('\u2019', "'").replace('\u2018', "'")
        t = t.replace('\u201c', '"').replace('\u201d', '"')
        t = t.replace('\u2014', '-').replace('\u2013', '-')
        # Remove HTML tags if present
        t = re.sub(r'<[^>]+>', '', t)
        return ' '.join(t.split())
    
    user_norm = normalize(user_quote)
    source_norm = normalize(source_text)
    
    # Check containment first
    if user_norm in source_norm or source_norm in user_norm:
        return 1.0
    
    # CRITICAL: autojunk=False prevents SequenceMatcher from ignoring common chars
    return SequenceMatcher(None, user_norm, source_norm, autojunk=False).ratio()


def compute_match_with_diffs(user_quote: str, source_text: str) -> tuple:
    """
    Computes match score AND identifies character-level differences.
    Returns: (score, diffs_list, verified_quote_html)
    """
    if not user_quote or not source_text:
        return 0.0, [], user_quote, ""
    
    # Strip HTML tags and decode entities from snippet
    clean_source = re.sub(r'<[^>]+>', '', source_text)
    clean_source = html.unescape(clean_source)
    
    # Light normalization for matching (preserve case for display)
    def normalize_for_match(t):
        t = t.replace('\u2019', "'").replace('\u2018', "'")
        t = t.replace('\u201c', '"').replace('\u201d', '"')
        t = t.replace('\u2014', '-').replace('\u2013', '-')
        return t
    
    user_norm = normalize_for_match(user_quote)
    source_norm = normalize_for_match(clean_source)
    
    # Use SequenceMatcher to find differences
    # CRITICAL: autojunk=False prevents SequenceMatcher from ignoring common chars
    matcher = SequenceMatcher(None, user_norm.lower(), source_norm.lower(), autojunk=False)
    score = matcher.ratio()
    
    diffs = []
    verified_html = ""
    
    # Get matching blocks and identify differences
    opcodes = matcher.get_opcodes()
    
    for tag, i1, i2, j1, j2 in opcodes:
        user_segment = user_norm[i1:i2]
        source_segment = source_norm[j1:j2]
        
        if tag == 'equal':
            # Matching text - no highlight
            verified_html += html.escape(user_segment)
        elif tag == 'replace':
            # Substitution - user wrote something different
            diffs.append(DiffSegment(
                position=i1,
                user_text=user_segment,
                source_text=source_segment,
                diff_type='substitution'
            ))
            verified_html += f'<span class="diff-error" title="Source: {html.escape(source_segment)}">{html.escape(user_segment)}</span>'
        elif tag == 'insert':
            # User added text not in source
            diffs.append(DiffSegment(
                position=i1,
                user_text=user_segment,
                source_text="",
                diff_type='insertion'
            ))
            verified_html += f'<span class="diff-error" title="Not in source">{html.escape(user_segment)}</span>'
        elif tag == 'delete':
            # User missing text that's in source
            diffs.append(DiffSegment(
                position=i1,
                user_text="",
                source_text=source_segment,
                diff_type='deletion'
            ))
            verified_html += f'<span class="diff-missing" title="Missing: {html.escape(source_segment)}">[...]</span>'
    
    return score, diffs, verified_html, source_norm


# =============================================================================
# FULL TEXT VERIFICATION (Extended Range)
# =============================================================================

async def fetch_full_opinion_text(cluster_id: str, client: httpx.AsyncClient, headers: Dict[str, str]) -> Optional[str]:
    """
    Fetch full opinion text from CourtListener API.
    
    Args:
        cluster_id: The cluster ID of the opinion
        client: httpx AsyncClient instance
        headers: Authorization headers
        
    Returns:
        Full plain text of opinion, or None if fetch fails
    """
    if not cluster_id:
        return None
    
    try:
        # First get the cluster to find the opinion ID
        cluster_url = f"{COURTLISTENER_BASE_URL}/clusters/{cluster_id}/"
        resp = await client.get(cluster_url, headers=headers)
        resp.raise_for_status()
        cluster_data = resp.json()
        
        # Get opinions list from cluster
        opinions = cluster_data.get("sub_opinions", [])
        if not opinions:
            logger.warning(f"No opinions found in cluster {cluster_id}")
            return None
        
        # Fetch the first (main) opinion's full text
        # Opinion URLs are like "/api/rest/v4/opinions/12345/"
        opinion_url = opinions[0] if isinstance(opinions[0], str) else opinions[0].get("resource_uri", "")
        if not opinion_url:
            return None
        
        # Make it absolute if relative
        if opinion_url.startswith("/"):
            opinion_url = f"https://www.courtlistener.com{opinion_url}"
        
        resp = await client.get(opinion_url, headers=headers)
        resp.raise_for_status()
        opinion_data = resp.json()
        
        # Try different text fields in order of preference
        full_text = (
            opinion_data.get("plain_text") or 
            opinion_data.get("html_with_citations") or
            opinion_data.get("html") or
            opinion_data.get("xml_harvard") or
            ""
        )
        
        # Strip HTML if present
        if full_text and ("<" in full_text):
            full_text = re.sub(r'<[^>]+>', ' ', full_text)
            full_text = html.unescape(full_text)
            full_text = ' '.join(full_text.split())
        
        logger.info(f"Fetched full text for cluster {cluster_id}: {len(full_text)} chars")
        return full_text if full_text else None
        
    except Exception as e:
        logger.warning(f"Failed to fetch full opinion text for cluster {cluster_id}: {e}")
        return None


def extract_extended_range(full_text: str, snippet: str, buffer: int = 200) -> Optional[str]:
    """
    DEPRECATED - Use find_quote_in_source() instead.
    
    Find snippet position in full text and extract extended range.
    
    Args:
        full_text: Complete opinion text
        snippet: The snippet returned by search API
        buffer: Characters to include before and after snippet
        
    Returns:
        Extended range (buffer + snippet + buffer), or None if snippet not found
    """
    if not full_text or not snippet:
        return None
    
    # Clean snippet for matching (remove HTML markup, normalize whitespace)
    clean_snippet = re.sub(r'<[^>]+>', '', snippet)
    clean_snippet = html.unescape(clean_snippet)
    clean_snippet = ' '.join(clean_snippet.split())
    
    # Normalize full text similarly
    clean_full = ' '.join(full_text.split())
    
    # Try exact match first
    pos = clean_full.find(clean_snippet)
    
    if pos == -1:
        # Try fuzzy location using first 50 chars of snippet
        search_fragment = clean_snippet[:50]
        pos = clean_full.find(search_fragment)
        
    if pos == -1:
        # Try with SequenceMatcher to find best match location
        # CRITICAL: autojunk=False prevents SequenceMatcher from ignoring common chars
        matcher = SequenceMatcher(None, clean_full.lower(), clean_snippet.lower(), autojunk=False)
        match = matcher.find_longest_match(0, len(clean_full), 0, len(clean_snippet))
        if match.size > 20:  # At least 20 chars matching
            pos = match.a
        else:
            logger.warning(f"Could not locate snippet in full text")
            return None
    
    # Extract extended range
    start = max(0, pos - buffer)
    end = min(len(clean_full), pos + len(clean_snippet) + buffer)
    
    extended = clean_full[start:end]
    logger.info(f"Extracted extended range: {len(extended)} chars (pos={pos}, buffer={buffer})")
    
    return extended


def find_quote_in_source(user_quote: str, full_text: str, buffer: int = 50) -> Optional[str]:
    """
    Find where user's quote best matches in full text and extract that region.
    
    This is the CORRECT approach: find where the USER's quote appears,
    not where the API snippet appears. This avoids false "deletion" diffs
    from buffer text the user never intended to quote.
    
    Args:
        user_quote: The user's submitted quotation
        full_text: Complete source text (opinion, article, etc.)
        buffer: Small buffer for edge comparison (default 50 chars)
        
    Returns:
        Source excerpt matching user's quote, or None if not found
    """
    if not user_quote or not full_text:
        return None
    
    # Normalize both texts for matching
    def normalize(t):
        t = re.sub(r'<[^>]+>', ' ', t)  # Strip HTML
        t = html.unescape(t)
        t = t.replace('\u2019', "'").replace('\u2018', "'")
        t = t.replace('\u201c', '"').replace('\u201d', '"')
        t = t.replace('\u2014', '-').replace('\u2013', '-')
        t = ' '.join(t.split())  # Normalize whitespace
        return t
    
    clean_quote = normalize(user_quote)
    clean_source = normalize(full_text)
    
    # Strategy 1: Try exact substring match first
    pos = clean_source.lower().find(clean_quote.lower())
    if pos != -1:
        # Found exact match - no buffer needed (avoids false diffs from context)
        start = pos
        end = pos + len(clean_quote)
        excerpt = clean_source[start:end]
        logger.info(f"Exact match found at pos={pos}, excerpt={len(excerpt)} chars")
        return excerpt
    
    # Strategy 2: Find best matching region using SequenceMatcher
    # CRITICAL: autojunk=False prevents SequenceMatcher from ignoring common chars
    matcher = SequenceMatcher(None, clean_source.lower(), clean_quote.lower(), autojunk=False)
    
    # Sum ALL matching blocks instead of just longest contiguous match
    # This handles quotes with small errors that break contiguous alignment
    matching_blocks = matcher.get_matching_blocks()
    total_matching = sum(block.size for block in matching_blocks)
    
    if total_matching < len(clean_quote) * 0.5:  # Less than 50% matching
        logger.warning(f"Could not locate quote in source (best match: {total_matching} chars)")
        return None
    
    # Find the first significant block to anchor position
    first_block = next((b for b in matching_blocks if b.size > 10), matching_blocks[0] if matching_blocks else None)
    if not first_block:
        return None
    
    # The block tells us where in source (block.a) corresponds to where in quote (block.b)
    # We want to extract region that covers the full quote length
    # Estimate start position: block.a - block.b (where quote would start if aligned)
    estimated_start = first_block.a - first_block.b
    estimated_start = max(0, estimated_start)
    estimated_end = estimated_start + len(clean_quote)
    estimated_end = min(len(clean_source), estimated_end)
    
    # Only add buffer if match quality is LOW (< 80%)
    # High-quality matches don't need buffer and it causes false "deletion" diffs
    match_ratio = total_matching / len(clean_quote)
    if match_ratio < 0.80:
        # Low quality match - add buffer to help alignment
        estimated_start = max(0, estimated_start - buffer)
        estimated_end = min(len(clean_source), estimated_end + buffer)
    
    excerpt = clean_source[estimated_start:estimated_end]
    logger.info(f"Fuzzy match: total_matching={total_matching}/{len(clean_quote)} chars, excerpt={len(excerpt)} chars, ratio={match_ratio:.0%}")
    
    return excerpt


def verify_against_full_text(
    user_quote: str, 
    snippet: str, 
    full_text: str, 
    buffer: int = 50
) -> tuple:
    """
    Verify user quote against full text source.
    
    IMPROVED LOGIC: Instead of finding snippet and adding buffer (which causes
    false deletion errors), we find where the USER's quote matches in the full
    text and extract just that region for comparison.
    
    Args:
        user_quote: The user's submitted quotation
        snippet: Original snippet from search API (fallback only)
        full_text: Complete opinion text
        buffer: Small buffer for edge comparison (default 50 chars)
        
    Returns:
        (score, diffs_list, verified_quote_html, source_quote)
    """
    if not full_text:
        # Fall back to snippet-only comparison
        return compute_match_with_diffs(user_quote, snippet)
    
    # Find user's quote in full text (the CORRECT approach)
    source_excerpt = find_quote_in_source(user_quote, full_text, buffer)
    
    if not source_excerpt:
        # Couldn't locate in full text - fall back to snippet
        logger.info("Could not locate quote in full text, using snippet")
        return compute_match_with_diffs(user_quote, snippet)
    
    # Compare user quote against the source excerpt
    return compute_match_with_diffs(user_quote, source_excerpt)


# Stop words for distinctiveness scoring
STOP_WORDS = {
    'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'by', 'at', 'and', 'or',
    'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'with',
    'as', 'from', 'are', 'not', 'but', 'if', 'then', 'than', 'so', 'no',
    'yes', 'all', 'any', 'each', 'which', 'who', 'whom', 'what', 'when',
    'where', 'why', 'how', 'out', 'said', 'says', 'told', 'asked', 'replied',
    # Common legal terms
    'court', 'case', 'plaintiff', 'defendant', 'action', 'held', 'order',
    'judgment', 'motion', 'filed', 'claim', 'party', 'parties',
}


def score_word_distinctiveness(word: str) -> int:
    """
    Score a word's distinctiveness for search anchor selection.
    Higher score = more distinctive = better search anchor.
    """
    word_lower = word.lower().strip()
    word_clean = re.sub(r'[^\w]', '', word_lower)
    
    if not word_clean or len(word_clean) < 2:
        return 0
    
    if word_lower in STOP_WORDS:
        return 0
    
    # Drug/chemical patterns
    drug_suffixes = ('pirone', 'prine', 'zepam', 'olan', 'etine', 'amine', 
                     'azole', 'mycin', 'cillin', 'statin', 'pril', 'sartan',
                     'olol', 'dipine', 'oxacin', 'cycline', 'dronate')
    if any(word_clean.endswith(suffix) for suffix in drug_suffixes):
        return 100
    
    # Legal citation symbols
    if '§' in word or word_lower in ('u.s.', 'm.r.s.', 'f.2d', 'f.3d', 'f.supp'):
        return 90
    
    # Numbers (statute references, years)
    if re.match(r'^\d+$', word_clean):
        if len(word_clean) == 4:  # Year
            return 85
        return 80
    
    # Long words are usually more distinctive
    if len(word_clean) >= 10:
        return 70
    
    if len(word_clean) >= 7:
        return 50
    
    # Capitalized words (proper nouns)
    if word and word[0].isupper() and len(word_clean) >= 3:
        return 40
    
    if len(word_clean) >= 3:
        return 20
    
    return 0


def extract_anchor_window(text: str, window_size: int = 40, min_score: int = 50) -> Optional[str]:
    """
    Extract a character window around the most distinctive word in text.
    
    Layer 1 of two-layer search: Creates a search string anchored to a
    distinctive word, avoiding typo-prone common words.
    
    Window positioning based on anchor location:
    - First 25% of text:  anchor + chars after
    - Middle 50% of text: chars before + anchor + chars after (centered)
    - Last 25% of text:   chars before + anchor
    
    Args:
        text: User's quote text
        window_size: Total characters to extract around anchor (default 40)
        min_score: Minimum distinctiveness score to qualify (default 50)
        
    Returns:
        Character window string, or None if no suitable anchor found
    """
    if not text or len(text) < 20:
        return None
    
    # Find all words with positions
    best_anchor = None
    best_score = -1
    best_start = 0
    best_end = 0
    
    for match in re.finditer(r'§\s*\d+|\d+\s*[A-Z]\.[A-Z]\.[A-Z]\.|\S+', text):
        word = match.group()
        score = score_word_distinctiveness(word)
        if score > best_score and score >= min_score:
            best_score = score
            best_anchor = word
            best_start = match.start()
            best_end = match.end()
    
    if not best_anchor:
        logger.info("No suitable anchor found (all words below min_score)")
        return None
    
    text_len = len(text)
    anchor_pos_ratio = best_start / text_len
    
    # Determine window boundaries based on anchor position
    if anchor_pos_ratio < 0.25:
        # Anchor near start: take anchor + chars after
        win_start = best_start
        win_end = min(text_len, best_end + window_size)
    elif anchor_pos_ratio > 0.75:
        # Anchor near end: take chars before + anchor
        win_start = max(0, best_start - window_size)
        win_end = best_end
    else:
        # Anchor in middle: center the window
        half_window = window_size // 2
        win_start = max(0, best_start - half_window)
        win_end = min(text_len, best_end + half_window)
    
    window = text[win_start:win_end].strip()
    
    logger.info(f"Anchor window: '{best_anchor}' (score={best_score}) at {anchor_pos_ratio:.0%} → '{window[:50]}...'")
    
    return window


def split_at_dashes(text: str) -> str:
    """
    Split text at em-dashes and return the segment with the most distinctive word.
    
    Em-dashes often join clauses without spaces ("basin—Seymour"), which creates
    tokens that won't match the source text. Like ellipsis handling, we split
    and take the best segment.
    
    Example:
        Input:  "One for the basin—Seymour stops taking the buspirone."
        Output: "Seymour stops taking the buspirone."  (has buspirone, score 100)
    """
    if not DASH_SPLIT_PATTERN.search(text):
        return text
    
    segments = DASH_SPLIT_PATTERN.split(text)
    segments = [s.strip() for s in segments if s.strip()]
    
    if not segments:
        return text
    
    if len(segments) == 1:
        return segments[0]
    
    # Find segment with highest distinctiveness score
    best_segment = segments[0]
    best_score = -1
    
    for seg in segments:
        # Score each word in segment, take max
        for match in re.finditer(r'\S+', seg):
            word = match.group()
            score = score_word_distinctiveness(word)
            if score > best_score:
                best_score = score
                best_segment = seg
    
    logger.info(f"Split at dash: {len(segments)} segments, best score={best_score}, selected: '{best_segment[:50]}...'")
    return best_segment


def extract_distinctive_window(text: str, max_chars: int = 200) -> str:
    """
    Extract a search window starting from the most distinctive word.
    Returns up to max_chars starting from the highest-scoring word's position.
    """
    if len(text) <= max_chars:
        return text
    
    # Tokenize while preserving positions
    words_with_pos = []
    for match in re.finditer(r'\S+', text):
        words_with_pos.append((match.group(), match.start(), match.end()))
    
    if not words_with_pos:
        return text[:max_chars]
    
    # Score each word
    best_score = -1
    best_pos = 0
    
    for word, start, end in words_with_pos:
        score = score_word_distinctiveness(word)
        if score > best_score:
            best_score = score
            best_pos = start
    
    # Extract window starting at best position
    window = text[best_pos:best_pos + max_chars]
    
    logging.getLogger(__name__).info(f"Distinctive window: score={best_score}, starts at char {best_pos}: '{window[:50]}...'")
    
    return window


def compute_dynamic_threshold(quote_len: int, snippet_len: int, base_threshold: float = 0.90) -> float:
    """
    Adjust match threshold based on length ratio.
    
    SequenceMatcher.ratio() = 2 * matches / (len_a + len_b)
    When snippet is shorter than quote, perfect overlap is capped.
    
    Example: quote=350, snippet=200, perfect overlap=200 chars
             max_ratio = 2*200 / (350+200) = 0.727
             
    We require base_threshold (90%) of what's theoretically achievable.
    """
    if snippet_len >= quote_len:
        return base_threshold
    
    # Max possible ratio for perfect overlap of shorter string
    max_possible = (2 * snippet_len) / (quote_len + snippet_len)
    
    # Require 90% of what's theoretically achievable
    adjusted = base_threshold * max_possible
    
    # Floor at 40% to avoid false positives
    return max(adjusted, 0.40)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="CourtListener Test App",
    description="Standalone test for CourtListener API integration",
    version="1.9.0"
)

# =============================================================================
# REQUEST MODELS
# =============================================================================

class SearchRequest(BaseModel):
    quote: str
    limit: int = 5

class CitationRequest(BaseModel):
    citation: str

# =============================================================================
# NER-BASED KEYWORD EXTRACTION
# =============================================================================

def extract_keywords_ner(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract distinctive keywords using NER + fallback.
    
    Priority:
    1. Named entities (PERSON, ORG, GPE, FAC, PRODUCT, EVENT, WORK_OF_ART)
    2. Dates and numbers (especially years)
    3. Distinctive nouns (filtered by stop words)
    
    Returns list of keywords for search query.
    """
    keywords = []
    seen = set()
    
    # Stop words for fallback extraction
    stop_words = {
        # Common English
        'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'by', 'at', 'and', 'or', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'with', 'as', 'from', 'are', 'not', 'but', 'if', 'then', 'than', 'so', 'no', 'yes', 'all', 'any', 'each', 'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how', 'out', 'against', 'into', 'upon', 'under', 'over', 'between', 'through', 'during', 'before', 'after', 'above', 'below', 'such', 'other', 'same', 'only', 'own', 'more', 'most', 'some', 'also', 'just', 'even', 'both', 'either', 'neither', 'whether', 'while', 'although', 'because', 'since', 'unless', 'until', 'however', 'therefore', 'thus', 'hence', 'there', 'here', 'now', 'still', 'yet', 'already', 'always', 'never', 'ever', 'often', 'sometimes', 'usually', 'again', 'further', 'once', 'twice',
        # Common legal terms (appear in nearly every case)
        'action', 'commenced', 'plaintiff', 'defendant', 'appellee', 'appellant', 'court', 'case', 'matter', 'cause', 'suit', 'claim', 'filed', 'brought', 'sued', 'recover', 'damages', 'judgment', 'order', 'decree', 'held', 'found', 'decided', 'ruled', 'affirmed', 'reversed', 'remanded', 'denied', 'granted', 'motion', 'petition', 'complaint', 'answer', 'issue', 'question', 'fact', 'law', 'evidence', 'testimony', 'witness', 'trial', 'hearing', 'proceeding', 'party', 'parties', 'person', 'persons', 'belonging', 'said', 'made', 'given', 'done', 'taken'
    }
    
    def add_keyword(word: str):
        """Add keyword if not duplicate."""
        word_lower = word.lower().strip()
        if word_lower and word_lower not in seen and len(word_lower) > 2:
            keywords.append(word_lower)
            seen.add(word_lower)
    
    # Try NER extraction first
    if NER_AVAILABLE and nlp:
        try:
            doc = nlp(text[:500])  # Limit text length for performance
            
            # Priority 1: Named entities
            priority_labels = {'PERSON', 'ORG', 'GPE', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LOC'}
            for ent in doc.ents:
                if ent.label_ in priority_labels:
                    # Add each word of multi-word entities
                    for word in ent.text.split():
                        if word.lower() not in stop_words:
                            add_keyword(word)
            
            # Priority 2: Dates and numbers (especially 4-digit years)
            for ent in doc.ents:
                if ent.label_ in {'DATE', 'CARDINAL', 'MONEY', 'QUANTITY'}:
                    # Extract just numbers/years
                    numbers = re.findall(r'\b\d{4}\b|\b\d+\b', ent.text)
                    for num in numbers:
                        add_keyword(num)
            
            # Priority 3: Concrete nouns (not in stop words)
            for token in doc:
                if token.pos_ in {'NOUN', 'PROPN'} and token.text.lower() not in stop_words:
                    add_keyword(token.text)
                    
            logger.info(f"NER extracted {len(keywords)} keywords: {keywords[:10]}")
            
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}, falling back to regex")
            keywords = []
            seen = set()
    
    # Fallback: regex-based extraction if NER didn't produce enough
    if len(keywords) < max_keywords:
        # Extract 4-digit years first (very distinctive)
        years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
        for year in years:
            add_keyword(year)
        
        # Extract remaining distinctive words
        text_clean = re.sub(r'[^\w\s]', ' ', text[:300].lower())
        words = text_clean.split()
        for word in words:
            if word not in stop_words and len(word) > 2:
                add_keyword(word)
    
    result = keywords[:max_keywords]
    logger.info(f"Final keywords ({len(result)}): {result}")
    return result

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _parse_search_result(item: Dict[str, Any], quote_text: str, trusted: bool = False, full_text: str = "") -> SearchResult:
    """Parse a CourtListener search result item into SearchResult.
    
    Args:
        item: Raw search result from API
        quote_text: User's submitted quotation
        trusted: Whether this is from exact phrase match
        full_text: Optional full opinion text for extended verification
    """
    cluster_id = str(item.get("cluster_id", ""))
    court_raw = item.get("court", "")
    court_parts = court_raw.split("/") if court_raw else []
    court_id = court_parts[-2] if len(court_parts) >= 2 else (court_parts[0] if court_parts else "")
    court_name = COURT_NAME_MAP.get(court_id, court_id)
    
    # Get citation
    citation = ""
    if item.get("citation"):
        citation = item.get("citation", [""])[0] if isinstance(item.get("citation"), list) else item.get("citation", "")
    
    # Get snippet from highlights or text
    snippet = ""
    if item.get("snippet"):
        snippet = item.get("snippet", "")
    elif item.get("text"):
        snippet = item.get("text", "")[:500]
    
    # FULL TEXT VERIFICATION: Use extended range if full text available
    # This catches errors outside the original snippet (e.g., "patent" vs "patently")
    if full_text:
        computed_score, diff_objects, verified_quote, source_quote = verify_against_full_text(
            quote_text, snippet, full_text, buffer=50
        )
        logger.info(f"Full text verification: score={computed_score:.2f}, diffs={len(diff_objects)}")
    else:
        # Fall back to snippet-only comparison
        computed_score, diff_objects, verified_quote, source_quote = compute_match_with_diffs(quote_text, snippet)
    
    diffs = [asdict(d) for d in diff_objects]
    
    # For trusted matches (exact phrase found), use 1.0 as base score
    # But if diffs were detected, the score reflects actual similarity
    if trusted and not diffs:
        match_score = 1.0
    else:
        match_score = computed_score
    
    return SearchResult(
        success=True,
        case_name=item.get("caseName", item.get("case_name", "")),
        citation=citation,
        court=court_name,
        date_filed=item.get("dateFiled", item.get("date_filed", "")),
        snippet=snippet,
        url=f"https://www.courtlistener.com/opinion/{cluster_id}/",
        cluster_id=cluster_id,
        match_score=match_score,
        diffs=diffs,
        verified_quote=verified_quote,
        source_quote=source_quote
    )


async def search_by_quote(quote_text: str, limit: int = 5) -> SearchResponse:
    """
    Search CourtListener for opinions containing quote.
    Returns SearchResponse with trace and best available results (with error highlighting).
    
    4-phase search strategy:
    0. Distinctive anchors only (case identification - typo resistant)
    1. Distinctive window (200 chars from most distinctive word)
    2. Fuzzy matching (returns results with diff highlighting)
    3. Keyword fallback (returns results with diff highlighting)
    """
    trace = []
    logger.info(f"search_by_quote called with: '{quote_text[:50]}...'")
    logger.info(f"API Key configured: {bool(ACTIVE_API_KEY)} (source: {API_KEY_SOURCE})")
    
    if not ACTIVE_API_KEY:
        logger.error("No API key configured!")
        trace.append("❌ API Key not configured")
        return SearchResponse(results=[SearchResult(
            success=False, case_name="", citation="", court="",
            date_filed="", snippet="", url="", cluster_id="",
            error="No API key configured. Set COURTLISTENER_API_KEY or CL_API_KEY"
        )], trace=trace)
    
    headers = {
        "Authorization": f"Token {ACTIVE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Clean quote using NFC normalization (preserves §, ¶)
    clean_q = clean_quote_text(quote_text)
    
    # Split at em-dashes and take segment with most distinctive word
    clean_q = split_at_dashes(clean_q)
    
    # Extract 40-char window around best anchor for Layer 1 (case identification)
    anchor_window = extract_anchor_window(clean_q, window_size=40, min_score=50)
    
    # Extract 200-char window starting from most distinctive word
    distinctive_q = extract_distinctive_window(clean_q, max_chars=200)
    
    # Also prepare shorter fragment for fallback
    words = distinctive_q.split()
    short_q = " ".join(words[:15]) if len(words) > 15 else distinctive_q
    
    trace.append(f"Search window: {distinctive_q[:60]}...")
    logger.info(f"Search window: {distinctive_q[:60]}...")
    
    search_url = f"{COURTLISTENER_BASE_URL}/search/"
    
    # Track best results found across all phases (for showing with errors)
    best_results = []
    best_items = []  # Raw API items for re-verification with full text
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            
            # =================================================================
            # PHASE 0: Anchor window search (case identification - typo resistant)
            # =================================================================
            # Layer 1 of two-layer search: Search using 40-char window around
            # the most distinctive word. This avoids typos in common words
            # while providing enough context to identify the specific case.
            if anchor_window:
                trace.append(f"Phase 0 - Anchor window: {anchor_window[:50]}...")
                logger.info(f"Phase 0 - Trying anchor window: {anchor_window}")
                
                # Search without quotes for fuzzy matching
                params = {
                    "q": anchor_window,
                    "type": "o",
                    "order_by": "score desc",
                    "page_size": 10
                }
                
                response = await client.get(search_url, headers=headers, params=params)
                
                if response.status_code == 401:
                    logger.error("401 Unauthorized - API key invalid")
                    trace.append("❌ 401 Unauthorized - Check API key")
                    return SearchResponse(results=[SearchResult(
                        success=False, case_name="", citation="", court="",
                        date_filed="", snippet="", url="", cluster_id="",
                        error="401 Unauthorized - Check API key"
                    )], trace=trace)
                
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                
                if items:
                    trace.append(f"Phase 0 - Found {len(items)} candidate(s), verifying...")
                    logger.info(f"Phase 0 - Found {len(items)} candidates")
                    
                    # Try each candidate until we find one containing the quote
                    for item in items[:5]:  # Check top 5 candidates
                        cluster_id = str(item.get("cluster_id", ""))
                        case_name = item.get("caseName", item.get("case_name", ""))[:40]
                        
                        full_text = await fetch_full_opinion_text(cluster_id, client, headers)
                        if not full_text:
                            continue
                        
                        # Layer 2: Verify quote exists in this case using fuzzy match
                        source_excerpt = find_quote_in_source(clean_q, full_text, buffer=50)
                        
                        if source_excerpt:
                            # Found it! Now do full verification
                            trace.append(f"✓ Quote found in: {case_name}...")
                            logger.info(f"Phase 0 - Quote verified in cluster {cluster_id}")
                            
                            result = _parse_search_result(item, quote_text, trusted=False, full_text=full_text)
                            
                            # Accept if match score >= 80%
                            if result.match_score >= 0.80:
                                trace.append(f"✅ Phase 0 verified: {result.match_score:.0%} match")
                                logger.info(f"✅ Phase 0 success: {result.match_score:.0%} match")
                                return SearchResponse(results=[result], trace=trace)
                            else:
                                trace.append(f"   Score {result.match_score:.0%} < 80%, checking next candidate...")
                        else:
                            trace.append(f"   {case_name}... quote not found")
                    
                    trace.append("Phase 0 - No candidate passed verification")
                    logger.info("Phase 0 - No candidate passed 80% threshold")
                else:
                    trace.append("Phase 0 - No candidates found")
                    logger.info("Phase 0 - 0 results from anchor window search")
            
            # =================================================================
            # PHASE 1: Exact phrase strategies (trusted, match_score = 1.0)
            # =================================================================
            exact_strategies = [
                {"name": "Distinctive Window", "q": distinctive_q},
                {"name": "Short Fragment", "q": short_q},
            ]
            
            for strategy in exact_strategies:
                query = f'"{strategy["q"]}"'  # Exact phrase matching
                trace.append(f"Trying: {strategy['name']}...")
                logger.info(f"Phase 1 - Trying: {strategy['name']}...")
                
                params = {
                    "q": query,
                    "type": "o",
                    "order_by": "dateFiled asc",
                    "page_size": limit
                }
                
                response = await client.get(search_url, headers=headers, params=params)
                
                if response.status_code == 401:
                    logger.error("401 Unauthorized - API key invalid")
                    trace.append("❌ 401 Unauthorized - Check API key")
                    return SearchResponse(results=[SearchResult(
                        success=False, case_name="", citation="", court="",
                        date_filed="", snippet="", url="", cluster_id="",
                        error="401 Unauthorized - Check API key"
                    )], trace=trace)
                
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                
                if items:
                    # Fetch full text for verification (use first result's cluster)
                    trace.append("Fetching full text for verification...")
                    first_cluster_id = str(items[0].get("cluster_id", ""))
                    full_text = await fetch_full_opinion_text(first_cluster_id, client, headers)
                    
                    if full_text:
                        trace.append(f"✓ Full text retrieved ({len(full_text)} chars)")
                    else:
                        trace.append("⚠ Full text unavailable, using snippet only")
                    
                    results = [_parse_search_result(item, quote_text, trusted=True, full_text=full_text or "") for item in items[:limit]]
                    trace.append(f"✅ Found {len(results)} result(s) via exact phrase match")
                    logger.info(f"✅ Phase 1 ({strategy['name']}): Found {len(results)} via exact phrase")
                    return SearchResponse(results=results, trace=trace)
                else:
                    trace.append("❌ 0 results")
                    logger.info(f"❌ Phase 1 ({strategy['name']}): 0 results")
            
            # =================================================================
            # PHASE 2: Fuzzy strategies (no quotes, show all with diff highlighting)
            # =================================================================
            trace.append("Exact phrase exhausted, trying fuzzy matching...")
            logger.info("Phase 2 - Trying fuzzy matching...")
            
            fuzzy_strategies = [
                {"name": "Fuzzy Distinctive Window", "q": distinctive_q},
                {"name": "Fuzzy Short Fragment", "q": short_q},
            ]
            
            for strategy in fuzzy_strategies:
                search_query = strategy["q"]  # NO quotes = fuzzy matching
                trace.append(f"Trying: {strategy['name']}...")
                logger.info(f"Phase 2 - Trying: {strategy['name']}...")
                
                params = {
                    "q": search_query,
                    "type": "o",
                    "order_by": "dateFiled asc",
                    "page_size": 20  # Cast wider net
                }
                
                response = await client.get(search_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                
                if items:
                    # First pass: quick scoring without full text
                    parsed = [_parse_search_result(item, quote_text, trusted=False) for item in items]
                    # Sort by match score descending
                    parsed.sort(key=lambda r: r.match_score, reverse=True)
                    
                    # Check for high-confidence matches
                    quote_len = len(quote_text)
                    high_confidence = []
                    for r in parsed:
                        snippet_len = len(r.snippet) if r.snippet else 0
                        threshold = compute_dynamic_threshold(quote_len, snippet_len)
                        if r.match_score >= threshold:
                            high_confidence.append(r)
                            trace.append(f"   ↳ '{r.case_name[:30]}...' score={r.match_score:.2f} >= threshold={threshold:.2f}")
                    
                    if high_confidence:
                        # Fetch full text and re-verify the top match
                        trace.append("Fetching full text for verification...")
                        top_cluster_id = high_confidence[0].cluster_id
                        full_text = await fetch_full_opinion_text(top_cluster_id, client, headers)
                        
                        if full_text:
                            trace.append(f"✓ Full text retrieved ({len(full_text)} chars)")
                            # Re-parse with full text verification
                            top_item = items[0]  # Re-verify first item
                            verified_result = _parse_search_result(top_item, quote_text, trusted=False, full_text=full_text)
                            high_confidence[0] = verified_result
                        else:
                            trace.append("⚠ Full text unavailable, using snippet only")
                        
                        trace.append(f"✅ Found {len(high_confidence)} result(s) above dynamic threshold")
                        logger.info(f"✅ Phase 2 ({strategy['name']}): Found {len(high_confidence)} above dynamic threshold")
                        return SearchResponse(results=high_confidence[:limit], trace=trace)
                    else:
                        # Keep best results for potential display with errors
                        if parsed and (not best_results or parsed[0].match_score > best_results[0].match_score):
                            best_results = parsed[:limit]
                            best_items = items[:limit]  # Save items for later re-verification
                        trace.append(f"❌ {len(parsed)} results but none above dynamic threshold (best: {parsed[0].match_score:.0%})")
                        logger.info(f"❌ Phase 2 ({strategy['name']}): {len(parsed)} results but none above dynamic threshold")
                else:
                    trace.append("❌ 0 results")
                    logger.info(f"❌ Phase 2 ({strategy['name']}): 0 results")
            
            # =================================================================
            # PHASE 3: Keyword fallback
            # =================================================================
            trace.append("Fuzzy matching exhausted, trying keyword fallback...")
            logger.info("Phase 3 - Trying keyword fallback...")
            
            keywords = extract_keywords_ner(quote_text, max_keywords=10)
            
            if keywords:
                keyword_query = ' '.join(keywords)
                trace.append(f"Keywords: {keyword_query}")
                logger.info(f"Keywords: {keyword_query}")
                
                params = {
                    "q": keyword_query,
                    "type": "o",
                    "order_by": "dateFiled asc",
                    "page_size": 30
                }
                
                response = await client.get(search_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                
                if items:
                    parsed = [_parse_search_result(item, quote_text, trusted=False) for item in items]
                    # Sort by match score descending
                    parsed.sort(key=lambda r: r.match_score, reverse=True)
                    
                    # Check for matches above lower threshold
                    quote_len = len(quote_text)
                    verified = []
                    for r in parsed:
                        snippet_len = len(r.snippet) if r.snippet else 0
                        threshold = compute_dynamic_threshold(quote_len, snippet_len, base_threshold=0.50)
                        if r.match_score >= threshold:
                            verified.append(r)
                    
                    if verified:
                        # Fetch full text and re-verify the top match
                        trace.append("Fetching full text for verification...")
                        top_cluster_id = verified[0].cluster_id
                        full_text = await fetch_full_opinion_text(top_cluster_id, client, headers)
                        
                        if full_text:
                            trace.append(f"✓ Full text retrieved ({len(full_text)} chars)")
                            # Re-parse with full text verification
                            verified_result = _parse_search_result(items[0], quote_text, trusted=False, full_text=full_text)
                            verified[0] = verified_result
                        else:
                            trace.append("⚠ Full text unavailable, using snippet only")
                        
                        trace.append(f"✅ Found {len(verified)} result(s) via keyword fallback")
                        logger.info(f"✅ Phase 3: Found {len(verified)} via keyword fallback")
                        return SearchResponse(results=verified[:limit], trace=trace)
                    else:
                        # Keep best results for potential display with errors
                        if parsed and (not best_results or parsed[0].match_score > best_results[0].match_score):
                            best_results = parsed[:limit]
                            best_items = items[:limit]
                        trace.append(f"❌ Keyword results below threshold (best: {parsed[0].match_score:.0%})")
                        logger.info("❌ Phase 3: Keyword results below dynamic threshold")
                else:
                    trace.append("❌ 0 keyword results")
                    logger.info("❌ Phase 3: 0 keyword results")
            
            # =================================================================
            # Return best available results (with full text verification)
            # =================================================================
            if best_results and best_items:
                trace.append("Fetching full text for best available match...")
                top_cluster_id = best_results[0].cluster_id
                full_text = await fetch_full_opinion_text(top_cluster_id, client, headers)
                
                if full_text:
                    trace.append(f"✓ Full text retrieved ({len(full_text)} chars)")
                    # Re-parse with full text verification
                    verified_result = _parse_search_result(best_items[0], quote_text, trusted=False, full_text=full_text)
                    best_results[0] = verified_result
                else:
                    trace.append("⚠ Full text unavailable, using snippet only")
                
                trace.append(f"⚠️ Returning best available matches (may contain errors)")
                logger.info(f"⚠️ Returning {len(best_results)} best available results with potential errors")
                return SearchResponse(results=best_results, trace=trace)
            elif best_results:
                trace.append(f"⚠️ Returning best available matches (may contain errors)")
                logger.info(f"⚠️ Returning {len(best_results)} best available results with potential errors")
                return SearchResponse(results=best_results, trace=trace)
            
            trace.append("⛔ All strategies exhausted - no results found")
            logger.info("⛔ All phases exhausted - no results found")
            return SearchResponse(results=[], trace=trace)
                
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        trace.append(f"❌ HTTP Error: {e.response.status_code}")
        return SearchResponse(results=[SearchResult(
            success=False, case_name="", citation="", court="",
            date_filed="", snippet="", url="", cluster_id="",
            error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        )], trace=trace)
    except Exception as e:
        logger.error(f"Search error: {type(e).__name__}: {e}")
        trace.append(f"❌ Error: {str(e)}")
        return SearchResponse(results=[SearchResult(
            success=False, case_name="", citation="", court="",
            date_filed="", snippet="", url="", cluster_id="",
            error=str(e)
        )], trace=trace)


async def lookup_by_citation(citation: str) -> SearchResult:
    """Lookup a specific case by citation."""
    logger.info(f"lookup_by_citation called with: '{citation}'")
    
    if not ACTIVE_API_KEY:
        return SearchResult(
            success=False,
            case_name="",
            citation=citation,
            court="",
            date_filed="",
            snippet="",
            url="",
            cluster_id="",
            error="No API key configured. Set COURTLISTENER_API_KEY or CL_API_KEY"
        )
    
    headers = {
        "Authorization": f"Token {ACTIVE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            # Search for the citation
            search_url = f"{COURTLISTENER_BASE_URL}/search/"
            params = {
                "q": citation,
                "type": "o",
                "order_by": "score desc",
                "page_size": 1
            }
            
            logger.info(f"Citation lookup: {search_url} with params {params}")
            
            response = await client.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                return SearchResult(
                    success=False,
                    case_name="",
                    citation=citation,
                    court="",
                    date_filed="",
                    snippet="",
                    url="",
                    cluster_id="",
                    error="Citation not found"
                )
            
            item = results[0]
            cluster_id = str(item.get("cluster_id", ""))
            court_raw = item.get("court", "")
            court_parts = court_raw.split("/") if court_raw else []
            court_id = court_parts[-2] if len(court_parts) >= 2 else (court_parts[0] if court_parts else "")
            
            return SearchResult(
                success=True,
                case_name=item.get("caseName", item.get("case_name", "")),
                citation=citation,
                court=COURT_NAME_MAP.get(court_id, court_id),
                date_filed=item.get("dateFiled", item.get("date_filed", "")),
                snippet=item.get("snippet", "")[:500],
                url=f"https://www.courtlistener.com/opinion/{cluster_id}/",
                cluster_id=cluster_id
            )
            
    except Exception as e:
        logger.error(f"Citation lookup error: {e}")
        return SearchResult(
            success=False,
            case_name="",
            citation=citation,
            court="",
            date_filed="",
            snippet="",
            url="",
            cluster_id="",
            error=str(e)
        )

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Web UI for testing."""
    api_status = "✅ Configured" if ACTIVE_API_KEY else "❌ Missing"
    api_preview = f"{ACTIVE_API_KEY[:8]}...{ACTIVE_API_KEY[-4:]}" if ACTIVE_API_KEY else "Not set"
    key_source = API_KEY_SOURCE if API_KEY_SOURCE else "None"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CourtListener Tester v1.9</title>
        <style>
            body {{ font-family: sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; }}
            .box {{ border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            .status {{ background: #f5f5f5; padding: 10px; border-radius: 6px; margin-bottom: 15px; font-size: 13px; }}
            .status.ok {{ background: #d4edda; }}
            .status.error {{ background: #f8d7da; }}
            .trace {{ background: #333; color: #0f0; padding: 10px; font-family: monospace; font-size: 12px; border-radius: 5px; margin-top: 10px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }}
            button {{ padding: 10px 15px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; margin-right: 8px; margin-bottom: 8px; }}
            button:hover {{ background: #0056b3; }}
            button.secondary {{ background: #666; }}
            input, textarea {{ width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box; }}
            .result {{ background: #f8f9fa; padding: 15px; margin-top: 15px; border-left: 4px solid #007bff; border-radius: 4px; }}
            .result.error {{ border-left-color: #dc3545; background: #fff5f5; }}
            .result-title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
            .verified-quote {{ background: #fff; padding: 12px; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; line-height: 1.6; }}
            .diff-error {{ background-color: #ffeb3b; font-weight: bold; padding: 1px 3px; border-radius: 2px; cursor: help; }}
            .diff-missing {{ background-color: #ff9800; color: white; font-weight: bold; padding: 1px 3px; border-radius: 2px; cursor: help; }}
            .error-summary {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin-top: 10px; }}
            .error-item {{ margin: 5px 0; padding: 5px; background: #fff; border-radius: 3px; font-family: monospace; font-size: 12px; }}
            .snippet-label {{ color: #666; font-size: 0.9em; margin-top: 10px; }}
            .snippet-text {{ font-style: italic; color: #555; font-size: 0.9em; }}
            .meta {{ color: #666; font-size: 13px; margin: 3px 0; }}
            h2 {{ margin-top: 25px; color: #333; }}
            .quick-tests {{ margin-top: 15px; }}
            /* Side-by-side comparison */
            .comparison-container {{ display: flex; gap: 15px; margin: 15px 0; }}
            .comparison-panel {{ flex: 1; background: #fff; border: 1px solid #ddd; border-radius: 6px; overflow: hidden; }}
            .comparison-header {{ padding: 10px 15px; font-weight: bold; font-size: 14px; border-bottom: 1px solid #ddd; }}
            .comparison-header.user {{ background: #e3f2fd; color: #1565c0; }}
            .comparison-header.source {{ background: #e8f5e9; color: #2e7d32; }}
            .comparison-body {{ padding: 15px; line-height: 1.8; font-size: 14px; max-height: 300px; overflow-y: auto; }}
            .comparison-body.user .diff-error {{ background-color: #ffeb3b; }}
            .comparison-body.source {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>⚖️ CourtListener Tester v1.9</h1>
        <p style="color:#666">Legal quote verification with error detection</p>
        
        <div class="status {'ok' if ACTIVE_API_KEY else 'error'}">
            <strong>API Status:</strong> {api_status} &nbsp;|&nbsp; 
            <strong>Source:</strong> <code>{key_source}</code> &nbsp;|&nbsp;
            <strong>Key:</strong> <code>{api_preview}</code>
        </div>
        
        <div class="box">
            <h3>Search by Quote</h3>
            <textarea id="quote" rows="4" placeholder="Enter a quote from a court opinion..."></textarea>
            <button onclick="searchQuote()">🔍 Verify Quote</button>
            <button onclick="prefillLoving()" class="secondary">Load: Loving</button>
            <button onclick="prefillBrown()" class="secondary">Load: Brown</button>
            <button onclick="prefillRoe()" class="secondary">Load: Roe</button>
        </div>
        
        <div class="box">
            <h3>Lookup by Citation</h3>
            <input type="text" id="citation" placeholder="e.g., 388 U.S. 1">
            <button onclick="lookupCitation()">📖 Lookup Citation</button>
        </div>
        
        <div id="output"></div>

        <script>
            function prefillLoving() {{
                document.getElementById('quote').value = 'There is patently no legitimate overriding purpose independent of invidious racial discrimination which justifies this classification.';
            }}
            function prefillBrown() {{
                document.getElementById('quote').value = 'We conclude that, in the field of public education, the doctrine of separate but equal has no place.';
            }}
            function prefillRoe() {{
                document.getElementById('quote').value = 'This right of privacy, whether it be founded in the Fourteenth Amendment concept of personal liberty';
            }}
            
            async function searchQuote() {{
                const quote = document.getElementById('quote').value;
                if (!quote) return alert('Enter a quote');
                const out = document.getElementById('output');
                
                out.innerHTML = '<p>🔍 Searching and verifying...</p>';
                
                try {{
                    const res = await fetch('/search', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{quote: quote, limit: 5}})
                    }}).then(r => r.json());
                    
                    displayResults(res.results, res.trace, quote);
                }} catch (e) {{
                    out.innerHTML = '<div class="result error"><h3>Error</h3><pre>' + e + '</pre></div>';
                }}
            }}
            
            async function lookupCitation() {{
                const citation = document.getElementById('citation').value;
                if (!citation) return alert('Enter a citation');
                const out = document.getElementById('output');
                
                out.innerHTML = '<p>📖 Looking up citation...</p>';
                
                try {{
                    const res = await fetch('/citation', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{citation: citation}})
                    }}).then(r => r.json());
                    
                    displayResults([res], [], '');
                }} catch (e) {{
                    out.innerHTML = '<div class="result error"><h3>Error</h3><pre>' + e + '</pre></div>';
                }}
            }}
            
            function displayResults(results, trace, originalQuote) {{
                let html = '';
                
                // Show trace first (like Google Books)
                if (trace && trace.length > 0) {{
                    html += '<h3>Search Trace:</h3>';
                    html += '<div class="trace">' + trace.join('\\n') + '</div>';
                }}
                
                html += '<h3>Results (' + results.length + ')</h3>';
                
                if (results.length === 0) {{
                    html += '<p>❌ No matches found after all attempts.</p>';
                }} else {{
                    results.forEach(r => {{
                        if (r.error) {{
                            html += '<div class="result error"><h3>Error</h3><p>' + r.error + '</p></div>';
                            return;
                        }}
                        if (!r.success) {{
                            html += '<div class="result"><p>No results found</p></div>';
                            return;
                        }}
                        
                        const hasErrors = r.diffs && r.diffs.length > 0;
                        const statusIcon = hasErrors ? '⚠️' : '✅';
                        const statusText = hasErrors ? 'Differences Detected' : 'Verified';
                        
                        html += '<div class="result">';
                        html += '<div class="result-title">' + statusIcon + ' ' + (r.case_name || 'Unknown Case') + '</div>';
                        html += '<p class="meta"><strong>Citation:</strong> ' + (r.citation || 'N/A') + '</p>';
                        html += '<p class="meta"><strong>Court:</strong> ' + (r.court || 'N/A') + '</p>';
                        html += '<p class="meta"><strong>Date:</strong> ' + (r.date_filed || 'N/A') + '</p>';
                        html += '<p class="meta"><strong>Match Score:</strong> ' + Math.round((r.match_score || 0) * 100) + '%</p>';
                        
                        // Side-by-side comparison
                        if (r.verified_quote && r.source_quote) {{
                            html += '<div class="comparison-container">';
                            
                            // User's quote (left panel)
                            html += '<div class="comparison-panel">';
                            html += '<div class="comparison-header user">📝 Your Quotation</div>';
                            html += '<div class="comparison-body user">' + r.verified_quote + '</div>';
                            html += '</div>';
                            
                            // Authentic source (right panel)
                            html += '<div class="comparison-panel">';
                            html += '<div class="comparison-header source">✓ Authentic Source</div>';
                            html += '<div class="comparison-body source">' + escapeHtml(r.source_quote) + '</div>';
                            html += '</div>';
                            
                            html += '</div>';
                        }} else if (r.verified_quote) {{
                            html += '<div style="margin-top:10px"><strong>Your Quotation (' + statusText + '):</strong></div>';
                            html += '<div class="verified-quote">' + r.verified_quote + '</div>';
                        }}
                        
                        if (hasErrors) {{
                            html += '<div class="error-summary">';
                            html += '<strong>📋 Detected Differences (' + r.diffs.length + '):</strong>';
                            r.diffs.forEach((d, i) => {{
                                let desc = '';
                                if (d.diff_type === 'substitution') {{
                                    desc = 'You wrote "<b>' + escapeHtml(d.user_text) + '</b>" → Source has "<b>' + escapeHtml(d.source_text) + '</b>"';
                                }} else if (d.diff_type === 'insertion') {{
                                    desc = '"<b>' + escapeHtml(d.user_text) + '</b>" not found in source';
                                }} else if (d.diff_type === 'deletion') {{
                                    desc = 'Missing from your quote: "<b>' + escapeHtml(d.source_text) + '</b>"';
                                }}
                                html += '<div class="error-item">' + (i+1) + '. ' + desc + '</div>';
                            }});
                            html += '</div>';
                        }}
                        
                        if (r.url) {{
                            html += '<p><a href="' + r.url + '" target="_blank">View on CourtListener →</a></p>';
                        }}
                        
                        html += '</div>';
                    }});
                }}
                
                document.getElementById('output').innerHTML = html;
            }}
            
            function escapeHtml(text) {{
                if (!text) return '';
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }}
        </script>
    </body>
    </html>
    """

@app.post("/search")
async def search_endpoint(request: SearchRequest) -> Dict[str, Any]:
    """Search by quote text. Returns results and trace."""
    response = await search_by_quote(request.quote, request.limit)
    return {
        "results": [asdict(r) for r in response.results],
        "trace": response.trace
    }

@app.post("/citation")
async def citation_endpoint(request: CitationRequest) -> Dict[str, Any]:
    """Lookup by citation."""
    result = await lookup_by_citation(request.citation)
    return asdict(result)

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "api_key_configured": bool(ACTIVE_API_KEY),
        "api_key_source": API_KEY_SOURCE,
        "api_key_length": len(ACTIVE_API_KEY)
    }

@app.get("/config")
async def config():
    """Show configuration status."""
    return {
        "active_api_key": f"{ACTIVE_API_KEY[:8]}...{ACTIVE_API_KEY[-4:]}" if ACTIVE_API_KEY else "NOT SET",
        "api_key_source": API_KEY_SOURCE,
        "COURTLISTENER_BASE_URL": COURTLISTENER_BASE_URL,
        "API_TIMEOUT": API_TIMEOUT,
        "env_vars_checked": ["COURTLISTENER_API_KEY", "CL_API_KEY"]
    }

# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("CourtListener Test App Starting")
    logger.info("=" * 60)
    logger.info(f"API Key Configured: {bool(ACTIVE_API_KEY)}")
    logger.info(f"API Key Source: {API_KEY_SOURCE}")
    logger.info(f"API Key Length: {len(ACTIVE_API_KEY)}")
    if ACTIVE_API_KEY:
        logger.info(f"API Key Preview: {ACTIVE_API_KEY[:8]}...{ACTIVE_API_KEY[-4:]}")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
