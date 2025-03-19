import streamlit as st
from src.synthesizer.rag_pipeline import rag_pipeline

def main():
    st.title("AI Research Assistant")
    query = st.text_input("Enter your research query:")
    if query:
        results = rag_pipeline(query)
        if results:
            doc_indices, papers = results
            st.write(f"Results for query: {query}")
            
            for index in doc_indices:
                paper = papers[index]
                # Create a container for each paper
                with st.container():
                    st.markdown("---")
                    # Display title as a header
                    st.markdown(f"### {paper['title']}")
                    
                    # Display authors and publication date
                    st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                    st.markdown(f"**Published:** {paper['published']}")
                    
                    # Display abstract
                    st.markdown("**Abstract:**")
                    st.write(paper['summary'])
                    
                    # Create arXiv link
                    arxiv_id = paper.get('id', '').split('/')[-1]
                    if arxiv_id:
                        arxiv_link = f"https://arxiv.org/abs/{arxiv_id}"
                        st.markdown(f"[Read full paper on arXiv]({arxiv_link})")
                    
                    st.markdown("---")
        else:
            st.write("Failed to fetch data.")

if __name__ == "__main__":
    main()