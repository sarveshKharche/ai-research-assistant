import requests
import time
import xml.etree.ElementTree as ET

def fetch_arxiv(query, max_results=100, retries=3, backoff_factor=1.0):
    url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:  # Too Many Requests
                wait_time = backoff_factor * (2 ** attempt)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error occurred: {http_err}")
                break
        except requests.exceptions.ConnectionError as conn_err:
            wait_time = backoff_factor * (2 ** attempt)
            print(f"Connection error occurred: {conn_err}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
            break
    return None

def parse_arxiv_response(response):
    root = ET.fromstring(response)
    namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('arxiv:entry', namespace)
    
    papers = []
    for entry in entries:
        paper = {
            'title': entry.find('arxiv:title', namespace).text,
            'summary': entry.find('arxiv:summary', namespace).text,
            'authors': [author.find('arxiv:name', namespace).text for author in entry.findall('arxiv:author', namespace)],
            'published': entry.find('arxiv:published', namespace).text
        }
        papers.append(paper)
    return papers

if __name__ == "__main__":
    query = "artificial intelligence"
    response = fetch_arxiv(query)
    if response:
        papers = parse_arxiv_response(response)
        for paper in papers:
            print(f"Title: {paper['title']}")
            print(f"Summary: {paper['summary']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Published: {paper['published']}")
            print("-" * 80)
    else:
        print("Failed to fetch data.")