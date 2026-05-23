"""
Web search tool for legal research.

Supports:
- Search legal databases for UAE laws
- Fetch current regulatory information
- Look up legal precedent and interpretations
- Integration with Tavily, SerpAPI, or direct search
"""

from typing import Dict, List, Optional
import json
import logging
from urllib.parse import quote_plus

logger = logging.getLogger("tool_web_search")


class WebSearchTool:
    """Web search integration for legal research.

    Design:
    - Provider-agnostic (Tavily, SerpAPI, or mock)
    - Cached results for repeated queries
    - UAE legal source prioritization
    - Fallback to mock results when API not configured
    """

    def __init__(self, api_key: str = "", provider: str = "tavily"):
        self.api_key = api_key
        self.provider = provider
        self._cache = {}

    def search(self, query: str, num_results: int = 5,
               source_type: str = "legal") -> List[Dict]:
        """Search the web for legal information.

        Args:
            query: Search query
            num_results: Number of results to return
            source_type: 'legal', 'news', 'general'

        Returns:
            List of search results with title, url, snippet
        """
        # Check cache
        cache_key = f"{query}:{num_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.provider == "tavily" and self.api_key:
            results = self._search_tavily(query, num_results)
        elif self.api_key:
            results = self._search_generic(query, num_results)
        else:
            results = self._mock_search(query)

        self._cache[cache_key] = results
        return results

    def search_uae_laws(self, query: str) -> List[Dict]:
        """Search specifically for UAE legal information."""
        uae_query = f"UAE law {query} site:government.ae OR site:moj.gov.ae"
        return self.search(uae_query, source_type="legal")

    def _search_tavily(self, query: str, num_results: int) -> List[Dict]:
        """Search using Tavily API."""
        try:
            import requests
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": f"UAE legal: {query}",
                    "max_results": num_results,
                    "search_depth": "advanced",
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", r.get("snippet", "")),
                    "source": "tavily",
                })
            return results
        except Exception as e:
            logger.warning(f"Tavily search error: {e}")
            return self._mock_search(query)

    def _search_generic(self, query: str, num_results: int) -> List[Dict]:
        """Generic search via external API."""
        try:
            import requests
            encoded = quote_plus(f"UAE legal {query}")
            # This is a placeholder for actual search API integration
            logger.info(f"Web search: {query}")
            return [{"title": "Search result placeholder",
                     "url": "", "snippet": f"Results for: {query}"}]
        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return self._mock_search(query)

    def _mock_search(self, query: str) -> List[Dict]:
        """Return mock results when API is not configured."""
        return [{
            "title": f"UAE Legal: {query[:60]}",
            "snippet": (
                f"Research results for '{query}'. "
                f"Configure a web search API key (Tavily/SerpAPI) "
                f"for live legal data retrieval."
            ),
            "url": "https://www.legalautomation.ae/research",
            "source": "mock",
        }]

    def get_cached_count(self) -> int:
        """Get number of cached searches."""
        return len(self._cache)
