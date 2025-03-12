import streamlit as st
from src.synthesizer.rag_pipeline import rag_pipeline

def main():
    st.title("AI Research Assistant")
    query = st.text_input("Enter your research query:")
    if query:
        results = rag_pipeline(query)
        if results:
            doc_indices, documents = results
            st.write(f"Results for query: {query}")
            for index in doc_indices:
                st.write(f"Document {index + 1}: {documents[index]}")
        else:
            st.write("Failed to fetch data.")

if __name__ == "__main__":
    main()