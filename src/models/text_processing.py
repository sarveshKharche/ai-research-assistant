from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=512):
    """
    Chunk the input text into smaller segments of specified chunk size.
    
    Args:
    text (str): The input text to be chunked.
    chunk_size (int): The size of each chunk.
    
    Returns:
    list: A list of text chunks.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_embeddings(text_chunks):
    """
    Generate embeddings for the given text chunks using a pre-trained SentenceTransformer model.
    
    Args:
    text_chunks (list): A list of text chunks.
    
    Returns:
    numpy.ndarray: An array of embeddings for the text chunks.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings

if __name__ == "__main__":
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
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)
    print(embeddings)