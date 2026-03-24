#!/usr/bin/env python3
"""
Web Search Skill - 使用 DuckDuckGo 进行无需 API Key 的搜索
"""

import sys
import json
from ddgs import DDGS


def search(query: str, max_results: int = 5) -> list[dict]:
    """使用 DuckDuckGo 搜索"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return [{"error": str(e), "query": query}]


def main():
    if len(sys.argv) < 2:
        print("Usage: python web_search.py <query>", file=sys.stderr)
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    results = search(query)
    
    # 输出为 JSON 格式
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
