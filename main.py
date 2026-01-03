#!/usr/bin/env python3
"""
CourtListener Standalone Test App
=================================

Isolated test environment for debugging CourtListener API integration.
Deploy on Railway to test independently of QuotationGenie.

Endpoints:
    GET  /           - Web UI for testing
    POST /search     - Search by quote text
    POST /citation   - Lookup by citation (e.g., "388 U.S. 1")
    GET  /health     - Health check
    GET  /config     - Show configuration status
"""

import os
import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

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

COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "")
# Also check CL_API_KEY as fallback
if not COURTLISTENER_API_KEY:
    COURTLISTENER_API_KEY = os.getenv("CL_API_KEY", "")

COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
API_TIMEOUT = 30.0

# =============================================================================
# DATA CLASSES
# =============================================================================

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
    error: str = ""

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
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="CourtListener Test App",
    description="Standalone test for CourtListener API integration",
    version="1.0.0"
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
# CORE FUNCTIONS
# =============================================================================

async def search_by_quote(quote_text: str, limit: int = 5) -> List[SearchResult]:
    """Search CourtListener for opinions containing quote."""
    results = []
    
    logger.info(f"search_by_quote called with: '{quote_text[:50]}...'")
    logger.info(f"COURTLISTENER_API_KEY set: {bool(COURTLISTENER_API_KEY)}")
    logger.info(f"COURTLISTENER_API_KEY length: {len(COURTLISTENER_API_KEY)}")
    
    if not COURTLISTENER_API_KEY:
        logger.error("No API key configured!")
        return [SearchResult(
            success=False,
            case_name="",
            citation="",
            court="",
            date_filed="",
            snippet="",
            url="",
            cluster_id="",
            error="COURTLISTENER_API_KEY not configured"
        )]
    
    headers = {
        "Authorization": f"Token {COURTLISTENER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use first 150 chars for search
    search_text = quote_text[:150].strip()
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            search_url = f"{COURTLISTENER_BASE_URL}/search/"
            params = {
                "q": f'"{search_text}"',
                "type": "o",  # Opinions
                "order_by": "score desc",
                "page_size": limit
            }
            
            logger.info(f"Making request to: {search_url}")
            logger.info(f"Params: {params}")
            
            response = await client.get(search_url, headers=headers, params=params)
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 401:
                logger.error("401 Unauthorized - API key invalid or expired")
                return [SearchResult(
                    success=False,
                    case_name="",
                    citation="",
                    court="",
                    date_filed="",
                    snippet="",
                    url="",
                    cluster_id="",
                    error=f"401 Unauthorized - Check API key"
                )]
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Response keys: {list(data.keys())}")
            logger.info(f"Result count: {data.get('count', 0)}")
            
            for item in data.get("results", [])[:limit]:
                cluster_id = str(item.get("cluster_id", ""))
                court_id = item.get("court", "").split("/")[-2] if item.get("court") else ""
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
                
                results.append(SearchResult(
                    success=True,
                    case_name=item.get("caseName", item.get("case_name", "")),
                    citation=citation,
                    court=court_name,
                    date_filed=item.get("dateFiled", item.get("date_filed", "")),
                    snippet=snippet,
                    url=f"https://www.courtlistener.com/opinion/{cluster_id}/",
                    cluster_id=cluster_id
                ))
                
                logger.info(f"Found: {item.get('caseName', 'Unknown')} - {citation}")
            
            if not results:
                logger.info("No results found in search")
                
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        results.append(SearchResult(
            success=False,
            case_name="",
            citation="",
            court="",
            date_filed="",
            snippet="",
            url="",
            cluster_id="",
            error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        ))
    except Exception as e:
        logger.error(f"Search error: {type(e).__name__}: {e}")
        results.append(SearchResult(
            success=False,
            case_name="",
            citation="",
            court="",
            date_filed="",
            snippet="",
            url="",
            cluster_id="",
            error=str(e)
        ))
    
    return results


async def lookup_by_citation(citation: str) -> SearchResult:
    """Lookup a specific case by citation."""
    logger.info(f"lookup_by_citation called with: '{citation}'")
    
    if not COURTLISTENER_API_KEY:
        return SearchResult(
            success=False,
            case_name="",
            citation=citation,
            court="",
            date_filed="",
            snippet="",
            url="",
            cluster_id="",
            error="COURTLISTENER_API_KEY not configured"
        )
    
    headers = {
        "Authorization": f"Token {COURTLISTENER_API_KEY}",
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
            court_id = item.get("court", "").split("/")[-2] if item.get("court") else ""
            
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
    api_status = "✅ Configured" if COURTLISTENER_API_KEY else "❌ Missing"
    api_preview = f"{COURTLISTENER_API_KEY[:8]}...{COURTLISTENER_API_KEY[-4:]}" if COURTLISTENER_API_KEY else "Not set"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CourtListener Test App</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .status {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .status.ok {{ background: #d4edda; }}
            .status.error {{ background: #f8d7da; }}
            textarea {{ width: 100%; height: 100px; font-size: 14px; padding: 10px; }}
            input[type="text"] {{ width: 100%; padding: 10px; font-size: 14px; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 10px 5px 10px 0; }}
            button:hover {{ background: #0056b3; }}
            .result {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #007bff; }}
            .result.error {{ border-left-color: #dc3545; background: #fff5f5; }}
            .result h3 {{ margin: 0 0 10px 0; }}
            .meta {{ color: #666; font-size: 13px; }}
            pre {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; overflow-x: auto; }}
            #results {{ margin-top: 20px; }}
            .loading {{ color: #666; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>🔍 CourtListener Test App</h1>
        
        <div class="status {'ok' if COURTLISTENER_API_KEY else 'error'}">
            <strong>API Key Status:</strong> {api_status}<br>
            <strong>Key Preview:</strong> <code>{api_preview}</code>
        </div>
        
        <h2>Search by Quote</h2>
        <p>Enter a quote from a court opinion:</p>
        <textarea id="quote" placeholder="There is patently no legitimate overriding purpose independent of invidious racial discrimination which justifies this classification."></textarea>
        <br>
        <button onclick="searchQuote()">Search Quote</button>
        
        <h2>Lookup by Citation</h2>
        <p>Enter a case citation:</p>
        <input type="text" id="citation" placeholder="388 U.S. 1">
        <br>
        <button onclick="lookupCitation()">Lookup Citation</button>
        
        <h2>Quick Tests</h2>
        <button onclick="testLoving()">Test: Loving v. Virginia</button>
        <button onclick="testBrown()">Test: Brown v. Board</button>
        <button onclick="testRoe()">Test: Roe v. Wade</button>
        
        <div id="results"></div>
        
        <script>
            async function searchQuote() {{
                const quote = document.getElementById('quote').value;
                if (!quote) return alert('Enter a quote');
                
                document.getElementById('results').innerHTML = '<p class="loading">Searching...</p>';
                
                try {{
                    const response = await fetch('/search', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{quote: quote, limit: 5}})
                    }});
                    const data = await response.json();
                    displayResults(data);
                }} catch (e) {{
                    document.getElementById('results').innerHTML = '<div class="result error"><h3>Error</h3><pre>' + e + '</pre></div>';
                }}
            }}
            
            async function lookupCitation() {{
                const citation = document.getElementById('citation').value;
                if (!citation) return alert('Enter a citation');
                
                document.getElementById('results').innerHTML = '<p class="loading">Looking up...</p>';
                
                try {{
                    const response = await fetch('/citation', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{citation: citation}})
                    }});
                    const data = await response.json();
                    displayResults([data]);
                }} catch (e) {{
                    document.getElementById('results').innerHTML = '<div class="result error"><h3>Error</h3><pre>' + e + '</pre></div>';
                }}
            }}
            
            function testLoving() {{
                document.getElementById('quote').value = 'There is patently no legitimate overriding purpose independent of invidious racial discrimination which justifies this classification.';
                searchQuote();
            }}
            
            function testBrown() {{
                document.getElementById('quote').value = 'We conclude that, in the field of public education, the doctrine of separate but equal has no place.';
                searchQuote();
            }}
            
            function testRoe() {{
                document.getElementById('quote').value = 'This right of privacy, whether it be founded in the Fourteenth Amendment concept of personal liberty';
                searchQuote();
            }}
            
            function displayResults(results) {{
                let html = '<h3>Results (' + results.length + ')</h3>';
                
                for (const r of results) {{
                    if (r.error) {{
                        html += '<div class="result error"><h3>Error</h3><p>' + r.error + '</p></div>';
                    }} else if (r.success) {{
                        html += '<div class="result">';
                        html += '<h3>' + (r.case_name || 'Unknown Case') + '</h3>';
                        html += '<p class="meta"><strong>Citation:</strong> ' + (r.citation || 'N/A') + '</p>';
                        html += '<p class="meta"><strong>Court:</strong> ' + (r.court || 'N/A') + '</p>';
                        html += '<p class="meta"><strong>Date:</strong> ' + (r.date_filed || 'N/A') + '</p>';
                        if (r.snippet) html += '<p><em>' + r.snippet.substring(0, 300) + '...</em></p>';
                        if (r.url) html += '<p><a href="' + r.url + '" target="_blank">View on CourtListener →</a></p>';
                        html += '</div>';
                    }} else {{
                        html += '<div class="result"><p>No results found</p></div>';
                    }}
                }}
                
                html += '<h3>Raw Response</h3><pre>' + JSON.stringify(results, null, 2) + '</pre>';
                document.getElementById('results').innerHTML = html;
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
        "api_key_configured": bool(COURTLISTENER_API_KEY),
        "api_key_length": len(COURTLISTENER_API_KEY)
    }

@app.get("/config")
async def config():
    """Show configuration status."""
    return {
        "COURTLISTENER_API_KEY": f"{COURTLISTENER_API_KEY[:8]}...{COURTLISTENER_API_KEY[-4:]}" if COURTLISTENER_API_KEY else "NOT SET",
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
    logger.info(f"API Key Configured: {bool(COURTLISTENER_API_KEY)}")
    logger.info(f"API Key Length: {len(COURTLISTENER_API_KEY)}")
    if COURTLISTENER_API_KEY:
        logger.info(f"API Key Preview: {COURTLISTENER_API_KEY[:8]}...{COURTLISTENER_API_KEY[-4:]}")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
