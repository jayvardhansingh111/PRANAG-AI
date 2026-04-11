# orchestrator/research_fetcher.py
# Fetches scientific paper abstracts from Semantic Scholar and ArXiv,
# then uses the LLM to extract a single key finding from each abstract.

import json
import time
import requests
from typing import List, Optional
from JAY.shared.models import ResearchInsight
from JAY.shared.config import SEMANTIC_SCHOLAR_KEY, ARXIV_MAX_RESULTS


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def search_semantic_scholar(query: str, max_results: int = 5) -> List[dict]:
    """Search Semantic Scholar API. Returns raw paper dicts."""
    headers = {}
    if SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_KEY

    params = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,year,abstract,journal,externalIds"
    }
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, headers=headers, timeout=10
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.exceptions.Timeout:
        print("[RESEARCH] Semantic Scholar timeout.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[RESEARCH] Semantic Scholar error: {e}")
        return []


# ── ArXiv ─────────────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 3) -> List[dict]:
    """Search ArXiv API (no key required). Returns raw paper dicts."""
    import urllib.parse
    import xml.etree.ElementTree as ET

    params = {
        "search_query": f"all:{query}",
        "max_results":  max_results,
        "sortBy":       "relevance",
        "sortOrder":    "descending"
    }
    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    try:
        r   = requests.get(url, timeout=10)
        r.raise_for_status()
        ns  = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
        papers = []
        for entry in root.findall("atom:entry", ns):
            papers.append({
                "paper_id": entry.find("atom:id",      ns).text.strip(),
                "title":    entry.find("atom:title",   ns).text.strip(),
                "abstract": entry.find("atom:summary", ns).text.strip(),
                "year":     int(entry.find("atom:published", ns).text[:4]),
                "journal":  "arXiv preprint",
                "doi":      None
            })
        return papers
    except Exception as e:
        print(f"[RESEARCH] ArXiv error: {e}")
        return []


# ── Relevance Scorer ──────────────────────────────────────────────────────────

BOOST_TERMS = ["heat stress", "drought tolerance", "yield", "germination", "flowering"]

def score_relevance(query: str, title: str, abstract: str) -> float:
    """Keyword-based relevance score in [0, 1]."""
    query_words = set(query.lower().split())
    text        = (title + " " + abstract).lower()
    hits  = sum(1 for w in query_words if w in text)
    score = hits / max(len(query_words), 1)
    for term in BOOST_TERMS:
        if term in text:
            score = min(score + 0.1, 1.0)
    return round(score, 3)


# ── Key Finding Extractor ─────────────────────────────────────────────────────

def extract_key_finding(title: str, abstract: str, query: str) -> str:
    """Ask the LLM to extract one key finding from an abstract."""
    from .prompt_parser import call_llm
    prompt = (
        f"Research query: {query}\n\n"
        f"Paper title: {title}\n\n"
        f"Abstract: {abstract[:1000]}\n\n"
        "Extract ONE key finding directly relevant to the query. "
        "Write exactly 1-2 sentences with specific numbers if available. "
        "Return ONLY the finding text, no preamble."
    )
    try:
        result = call_llm(prompt, system="You are a scientific summarizer. Return only the key finding.")
        return result.strip().strip('"').strip("'")[:500]
    except Exception:
        sentences = abstract.split(". ")
        return sentences[0][:300] if sentences else "No finding extracted."


# ── Mock Data (offline fallback) ──────────────────────────────────────────────

MOCK_INSIGHTS = [
    {
        "paper_id":    "MOCK-2023-001",
        "title":       "Heat Stress Response in Wheat: Molecular Mechanisms",
        "year":        2023,
        "journal":     "Plant Cell & Environment",
        "key_finding": "Wheat varieties with HSP70 overexpression maintained 85% pollen viability at 45°C.",
        "relevance":   0.92,
        "doi":         "10.1111/pce.14567"
    },
    {
        "paper_id":    "MOCK-2024-002",
        "title":       "Grain Filling Under High Temperature in Triticum aestivum",
        "year":        2024,
        "journal":     "arXiv preprint",
        "key_finding": "Temperature above 35°C reduces starch accumulation by 40% and decreases grain weight.",
        "relevance":   0.87,
        "doi":         None
    }
]


# ── Main Entry Point ──────────────────────────────────────────────────────────

def fetch_research(query: str, max_results: int = 5) -> List[ResearchInsight]:
    """
    Fetch research from Semantic Scholar + ArXiv, score, extract findings.
    Falls back to MOCK_INSIGHTS when offline.
    """
    all_papers = []

    for paper in search_semantic_scholar(query, max_results):
        if paper.get("abstract"):
            all_papers.append({
                "paper_id": paper.get("paperId", "SS-unknown"),
                "title":    paper.get("title", "Unknown"),
                "abstract": paper.get("abstract", ""),
                "year":     paper.get("year", 2024),
                "journal":  (paper.get("journal") or {}).get("name"),
                "doi":      (paper.get("externalIds") or {}).get("DOI")
            })

    time.sleep(0.5)
    for paper in search_arxiv(query, max_results=3):
        all_papers.append(paper)

    if not all_papers:
        print("[RESEARCH] APIs unavailable — using mock insights.")
        return [ResearchInsight(**i) for i in MOCK_INSIGHTS]

    scored = [
        {**p, "relevance": score_relevance(query, p["title"], p.get("abstract", ""))}
        for p in all_papers
        if score_relevance(query, p["title"], p.get("abstract", "")) > 0.3
    ]
    scored.sort(key=lambda x: x["relevance"], reverse=True)

    insights = []
    for paper in scored[:max_results]:
        finding = extract_key_finding(paper["title"], paper.get("abstract", ""), query)
        insights.append(ResearchInsight(
            paper_id    = paper["paper_id"],
            title       = paper["title"],
            year        = paper["year"],
            journal     = paper.get("journal"),
            key_finding = finding,
            relevance   = paper["relevance"],
            doi         = paper.get("doi")
        ))

    return insights
