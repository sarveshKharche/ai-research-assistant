import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models.text_processing import chunk_text, generate_embeddings

def create_faiss_index(embeddings):
    """
    Create a FAISS index from the given embeddings.
    
    Args:
    embeddings (numpy.ndarray): An array of embeddings.
    
    Returns:
    faiss.Index: A FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, k=5):
    """
    Search the FAISS index with a query embedding.
    
    Args:
    index (faiss.Index): A FAISS index.
    query_embedding (numpy.ndarray): An embedding to search for.
    k (int): The number of nearest neighbors to return.
    
    Returns:
    tuple: Distances and indices of the nearest neighbors.
    """
    distances, indices = index.search(query_embedding, k)
    return distances, indices

if __name__ == "__main__":
    # Example text
    text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "in contrast to the natural intelligence displayed by humans and animals. "
        "Leading AI textbooks define the field as the study of 'intelligent agents': "
        "any device that perceives its environment and takes actions that maximize "
        "its chance of successfully achieving its goals. Colloquially, the term "
        "'artificial intelligence' is often used to describe machines (or computers) "
        "that mimic 'cognitive' functions that humans associate with the human mind, "
        "such as 'learning' and 'problem-solving'."
    )
    
    # Chunk text and generate embeddings
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)
    
    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Example query embedding
    query_text = "What is artificial intelligence?"
    query_chunks = chunk_text(query_text)
    query_embedding = generate_embeddings(query_chunks)
    
    # Search FAISS index
    distances, indices = search_faiss_index(index, query_embedding)
    
    print("Distances:", distances)
    print("Indices:", indices)