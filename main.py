#!/usr/bin/env python3
"""
CourtListener Standalone Test App (v1.6 - Error Detection)
============================================================

Isolated test environment for debugging CourtListener API integration.
Deploy on Railway to test independently of QuotationGenie.

Endpoints:
    GET  /           - Web UI for testing
    POST /search     - Search by quote text
    POST /citation   - Lookup by citation (e.g., "388 U.S. 1")
    GET  /health     - Health check
    GET  /config     - Show configuration status

Version History:
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
    
    return SequenceMatcher(None, user_norm, source_norm).ratio()


def compute_match_with_diffs(user_quote: str, source_text: str) -> tuple:
    """
    Computes match score AND identifies character-level differences.
    Returns: (score, diffs_list, verified_quote_html)
    """
    if not user_quote or not source_text:
        return 0.0, [], user_quote
    
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
    matcher = SequenceMatcher(None, user_norm.lower(), source_norm.lower())
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
    
    return score, diffs, verified_html


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
    version="1.6.0"
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

def _parse_search_result(item: Dict[str, Any], quote_text: str, trusted: bool = False) -> SearchResult:
    """Parse a CourtListener search result item into SearchResult."""
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
    
    # Compute match score with diff detection
    if trusted:
        match_score = 1.0
        diffs = []
        verified_quote = quote_text
    else:
        match_score, diff_objects, verified_quote = compute_match_with_diffs(quote_text, snippet)
        # Convert DiffSegment objects to dicts for JSON serialization
        diffs = [asdict(d) for d in diff_objects]
    
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
        verified_quote=verified_quote
    )


async def search_by_quote(quote_text: str, limit: int = 5) -> List[SearchResult]:
    """
    Search CourtListener for opinions containing quote.
    
    3-phase search strategy:
    1. Distinctive window (200 chars from most distinctive word)
    2. Fuzzy matching (90% threshold)
    3. Keyword fallback (50% threshold)
    """
    logger.info(f"search_by_quote called with: '{quote_text[:50]}...'")
    logger.info(f"API Key configured: {bool(ACTIVE_API_KEY)} (source: {API_KEY_SOURCE})")
    
    if not ACTIVE_API_KEY:
        logger.error("No API key configured!")
        return [SearchResult(
            success=False, case_name="", citation="", court="",
            date_filed="", snippet="", url="", cluster_id="",
            error="No API key configured. Set COURTLISTENER_API_KEY or CL_API_KEY"
        )]
    
    headers = {
        "Authorization": f"Token {ACTIVE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Clean quote using NFC normalization (preserves §, ¶)
    clean_q = clean_quote_text(quote_text)
    
    # Split at em-dashes and take segment with most distinctive word
    clean_q = split_at_dashes(clean_q)
    
    # Extract 200-char window starting from most distinctive word
    distinctive_q = extract_distinctive_window(clean_q, max_chars=200)
    
    # Also prepare shorter fragment for fallback
    words = distinctive_q.split()
    short_q = " ".join(words[:15]) if len(words) > 15 else distinctive_q
    
    logger.info(f"Search window: {distinctive_q[:60]}...")
    
    search_url = f"{COURTLISTENER_BASE_URL}/search/"
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            
            # =================================================================
            # PHASE 1: Exact phrase strategies (trusted, match_score = 1.0)
            # =================================================================
            exact_strategies = [
                {"name": "Distinctive Window", "q": distinctive_q},
                {"name": "Short Fragment", "q": short_q},
            ]
            
            for strategy in exact_strategies:
                query = f'"{strategy["q"]}"'  # Exact phrase matching
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
                    return [SearchResult(
                        success=False, case_name="", citation="", court="",
                        date_filed="", snippet="", url="", cluster_id="",
                        error="401 Unauthorized - Check API key"
                    )]
                
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                
                if items:
                    results = [_parse_search_result(item, quote_text, trusted=True) for item in items[:limit]]
                    logger.info(f"✅ Phase 1 ({strategy['name']}): Found {len(results)} via exact phrase")
                    return results
                else:
                    logger.info(f"❌ Phase 1 ({strategy['name']}): 0 results")
            
            # =================================================================
            # PHASE 2: Fuzzy strategies (no quotes, 90% threshold)
            # =================================================================
            logger.info("Phase 2 - Trying fuzzy matching...")
            
            fuzzy_strategies = [
                {"name": "Fuzzy Distinctive Window", "q": distinctive_q},
                {"name": "Fuzzy Short Fragment", "q": short_q},
            ]
            
            for strategy in fuzzy_strategies:
                search_query = strategy["q"]  # NO quotes = fuzzy matching
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
                    parsed = [_parse_search_result(item, quote_text, trusted=False) for item in items]
                    # Filter by DYNAMIC threshold (adjusts for length mismatch)
                    quote_len = len(quote_text)
                    verified = []
                    for r in parsed:
                        snippet_len = len(r.snippet) if r.snippet else 0
                        threshold = compute_dynamic_threshold(quote_len, snippet_len)
                        if r.match_score >= threshold:
                            verified.append(r)
                            logger.info(f"   ↳ '{r.case_name[:30]}...' score={r.match_score:.2f} >= threshold={threshold:.2f}")
                    
                    if verified:
                        logger.info(f"✅ Phase 2 ({strategy['name']}): Found {len(verified)} above dynamic threshold")
                        return verified[:limit]
                    else:
                        logger.info(f"❌ Phase 2 ({strategy['name']}): {len(parsed)} results but none above dynamic threshold")
                else:
                    logger.info(f"❌ Phase 2 ({strategy['name']}): 0 results")
            
            # =================================================================
            # PHASE 3: Keyword fallback (dynamic threshold, base 50%)
            # =================================================================
            logger.info("Phase 3 - Trying keyword fallback...")
            
            keywords = extract_keywords_ner(quote_text, max_keywords=10)
            
            if keywords:
                keyword_query = ' '.join(keywords)
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
                    # Dynamic threshold with lower base for keyword fallback
                    quote_len = len(quote_text)
                    verified = []
                    for r in parsed:
                        snippet_len = len(r.snippet) if r.snippet else 0
                        threshold = compute_dynamic_threshold(quote_len, snippet_len, base_threshold=0.50)
                        if r.match_score >= threshold:
                            verified.append(r)
                    
                    if verified:
                        logger.info(f"✅ Phase 3: Found {len(verified)} via keyword fallback")
                        return verified[:limit]
                    else:
                        logger.info("❌ Phase 3: Keyword results below dynamic threshold")
                else:
                    logger.info("❌ Phase 3: 0 keyword results")
            
            logger.info("⛔ All phases exhausted - no results found")
            return []
                
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return [SearchResult(
            success=False, case_name="", citation="", court="",
            date_filed="", snippet="", url="", cluster_id="",
            error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        )]
    except Exception as e:
        logger.error(f"Search error: {type(e).__name__}: {e}")
        return [SearchResult(
            success=False, case_name="", citation="", court="",
            date_filed="", snippet="", url="", cluster_id="",
            error=str(e)
        )]


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
        <title>CourtListener Tester v1.6</title>
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
        </style>
    </head>
    <body>
        <h1>⚖️ CourtListener Tester v1.6</h1>
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
                    
                    displayResults(res, quote);
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
                    
                    displayResults([res], '');
                }} catch (e) {{
                    out.innerHTML = '<div class="result error"><h3>Error</h3><pre>' + e + '</pre></div>';
                }}
            }}
            
            function displayResults(results, originalQuote) {{
                let html = '<h3>Results (' + results.length + ')</h3>';
                
                if (results.length === 0) {{
                    html += '<p>❌ No matches found.</p>';
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
                        
                        if (r.verified_quote) {{
                            html += '<div style="margin-top:10px"><strong>Your Quotation (' + statusText + '):</strong></div>';
                            html += '<div class="verified-quote">' + r.verified_quote + '</div>';
                        }}
                        
                        if (hasErrors) {{
                            html += '<div class="error-summary">';
                            html += '<strong>📋 Detected Differences (' + r.diffs.length + '):</strong>';
                            r.diffs.forEach((d, i) => {{
                                let desc = '';
                                if (d.diff_type === 'substitution') {{
                                    desc = 'You wrote "<b>' + d.user_text + '</b>" → Source has "<b>' + d.source_text + '</b>"';
                                }} else if (d.diff_type === 'insertion') {{
                                    desc = '"<b>' + d.user_text + '</b>" not found in source';
                                }} else if (d.diff_type === 'deletion') {{
                                    desc = 'Missing from your quote: "<b>' + d.source_text + '</b>"';
                                }}
                                html += '<div class="error-item">' + (i+1) + '. ' + desc + '</div>';
                            }});
                            html += '</div>';
                        }}
                        
                        if (r.snippet) {{
                            html += '<div class="snippet-label">Source snippet from CourtListener:</div>';
                            html += '<div class="snippet-text">"' + r.snippet.substring(0, 300) + '..."</div>';
                        }}
                        
                        if (r.url) {{
                            html += '<p><a href="' + r.url + '" target="_blank">View on CourtListener →</a></p>';
                        }}
                        
                        html += '</div>';
                    }});
                }}
                
                document.getElementById('output').innerHTML = html;
            }}
        </script>
    </body>
    </html>
    """

@app.post("/search")
async def search_endpoint(request: SearchRequest) -> List[Dict[str, Any]]:
    """Search by quote text."""
    results = await search_by_quote(request.quote, request.limit)
    return [asdict(r) for r in results]

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
