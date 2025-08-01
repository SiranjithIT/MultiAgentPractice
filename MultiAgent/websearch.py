from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from typing import List, Dict, Any
import re

class EnhancedWebSearch:
  def __init__(self):
    self.wrapper = DuckDuckGoSearchAPIWrapper(
      time="d",           # Last day for current info
      max_results=30,     
      region="wt-wt",     # Global results (wt-wt = worldwide)
      safesearch="moderate",
      source="text"       # Focus on text results
    )
    
    self.wrapper_week = DuckDuckGoSearchAPIWrapper(
      time="w", max_results=25, region="wt-wt", safesearch="moderate"
    )
    
    self.wrapper_month = DuckDuckGoSearchAPIWrapper(
      time="m", max_results=20, region="wt-wt", safesearch="moderate"
    )
    

    self.search = DuckDuckGoSearchResults(
      api_wrapper=self.wrapper, 
      output_format="list"
    )
    
    self.search_week = DuckDuckGoSearchResults(
      api_wrapper=self.wrapper_week, 
      output_format="list"
    )
    
    self.search_month = DuckDuckGoSearchResults(
      api_wrapper=self.wrapper_month, 
      output_format="list"
    )
  
  def enhance_query(self, query: str) -> List[str]:
    enhanced_queries = []
    enhanced_queries.append(query)
    if any(keyword in query.lower() for keyword in ['current', 'latest', 'recent', 'today', 'now', '2025']):
      enhanced_queries.append(f"{query} 2025")
      enhanced_queries.append(f"{query} latest news")
    
    if len(query.split()) > 2:
      enhanced_queries.append(f'"{query}"')
    
    
    if any(keyword in query.lower() for keyword in ['price', 'cost', 'stock']):
      enhanced_queries.append(f"{query} current price")
      enhanced_queries.append(f"{query} market")
    
    if any(keyword in query.lower() for keyword in ['weather', 'temperature']):
      enhanced_queries.append(f"{query} forecast")
      enhanced_queries.append(f"{query} today")
    
    if any(keyword in query.lower() for keyword in ['news', 'breaking']):
      enhanced_queries.append(f"{query} breaking news")
      enhanced_queries.append(f"{query} latest updates")
    
    return enhanced_queries[:3]
  
  def multi_timeframe_search(self, query: str) -> List[Dict[str, Any]]:
    all_results = []
    
    try:
      daily_results = self.search.invoke(query)
      if daily_results:
        for result in daily_results:
          result['timeframe'] = 'daily'
        all_results.extend(daily_results)
    except Exception as e:
      print(f"Daily search failed: {e}")
    
    try:
      weekly_results = self.search_week.invoke(query)
      if weekly_results:
        for result in weekly_results:
          result['timeframe'] = 'weekly'
        all_results.extend(weekly_results[:10])
    except Exception as e:
      print(f"Weekly search failed: {e}")
    
    return all_results
  
  def deep_search(self, query: str) -> List[Dict[str, Any]]:
    all_results = []
    seen_urls = set()
    enhanced_queries = self.enhance_query(query)
    
    for enhanced_query in enhanced_queries:
      try:
        results = self.multi_timeframe_search(enhanced_query)
        for result in results:
          if isinstance(result, dict) and 'link' in result:
            if result['link'] not in seen_urls:
              seen_urls.add(result['link'])
              result['query_variation'] = enhanced_query
              all_results.append(result)
          
      except Exception as e:
        print(f"Search failed for query '{enhanced_query}': {e}")
        continue
      
    all_results.sort(key=lambda x: (
        x.get('timeframe') == 'daily',  # Daily results first
        query.lower() in x.get('title', '').lower(),  # Title matches
        query.lower() in x.get('snippet', '').lower()  # Content matches
    ), reverse=True)
    
    return all_results[:25] 
  
  def filter_relevant_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not results:
      return []
    
    query_words = set(query.lower().split())
    filtered_results = []
    
    for result in results:
      if not isinstance(result, dict):
        continue
          
      title = result.get('title', '').lower()
      snippet = result.get('snippet', '').lower()
      relevance_score = 0
      
      title_words = set(title.split())
      title_overlap = len(query_words.intersection(title_words))
      relevance_score += title_overlap * 3
      
      snippet_words = set(snippet.split())
      snippet_overlap = len(query_words.intersection(snippet_words))
      relevance_score += snippet_overlap
      
      if query.lower() in title:
        relevance_score += 5
      if query.lower() in snippet:
        relevance_score += 3
      
      if relevance_score > 0:
        result['relevance_score'] = relevance_score
        filtered_results.append(result)
    
    filtered_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    return filtered_results
  
  def invoke(self, query: str) -> List[Dict[str, Any]]:
      """Main search method with enhanced capabilities"""
      print(f"Performing enhanced search for: {query}")
      raw_results = self.deep_search(query)
      filtered_results = self.filter_relevant_results(raw_results, query)
      print(f"Found {len(filtered_results)} relevant results")
      
      return filtered_results


search = EnhancedWebSearch()


if __name__ == "__main__":
  result = search.invoke("Gen AI")
  result.extend(search.invoke("Agent AI"))
  for i in result:
    print(i, end="\n\n")