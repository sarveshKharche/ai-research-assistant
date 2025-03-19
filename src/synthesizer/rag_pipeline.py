import numpy as np
from src.models.text_processing import chunk_text, generate_embeddings
from src.models.vector_store import create_faiss_index, search_faiss_index
from src.data.data_fetch import fetch_arxiv, parse_arxiv_response

def rag_pipeline(query):
    """
    Combine retrieval and generation steps.
    
    Args:
    query (str): The research query.
    
    Returns:
    tuple: Indices of the most relevant text chunks and the corresponding papers with full metadata.
    """
    # Fetch data from arXiv
    response = fetch_arxiv(query)
    if not response:
        return None
    
    papers = parse_arxiv_response(response)
    documents = [paper['summary'] for paper in papers]
    
    # Chunk documents and generate embeddings
    chunks = []
    doc_indices = []
    for i, doc in enumerate(documents):
        doc_chunks = chunk_text(doc)
        chunks.extend(doc_chunks)
        doc_indices.extend([i] * len(doc_chunks))
    
    embeddings = generate_embeddings(chunks)
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Generate query embedding
    query_embedding = generate_embeddings([query])
    
    # Search FAISS index
    distances, indices = search_faiss_index(index, query_embedding)
    
    # Map chunk indices to document indices
    doc_indices = [doc_indices[idx] for idx in indices[0]]
    
    return doc_indices, papers

if __name__ == "__main__":
    query = "What is artificial intelligence?"
    results = rag_pipeline(query)
    if results:
        doc_indices, papers = results
        for index in doc_indices:
            paper = papers[index]
            print(f"Title: {paper['title']}")
            print(f"Summary: {paper['summary']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Published: {paper['published']}")
            print("-" * 80)
    else:
        print("Failed to fetch data.")