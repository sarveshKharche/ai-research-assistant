# AI Research Assistant

## Project Overview

### Objective

Create an intelligent research assistant that automates the process of gathering, summarizing, and synthesizing academic literature or news articles. The goal is to help researchers quickly extract key insights from vast amounts of information.

## Key Components & Implementation Details

### 1. Integration with Scholarly APIs

#### Data Sources:

- **Scholarly Databases:** Use APIs like [arXiv API](https://arxiv.org/help/api/) to fetch metadata (titles, abstracts, authors, publication dates) and possibly full texts where available.

#### Implementation Tips:

- Build a modular data ingestion pipeline that can query multiple sources.
- Handle API rate limits and ensure robust error handling.

### 2. Retrieval Augmented Generation (RAG) & LangChain Integration

#### Data Processing:

- **Text Chunking:** Divide long articles or papers into manageable segments.
- **Embeddings:** Convert text segments into vector representations using models like Sentence Transformers.

#### Vector Store:

- Use FAISS to store and efficiently retrieve embeddings based on query similarity.

#### RAG Pipeline:

- **Retrieval:** Given a research query, retrieve the most relevant text chunks.
- **Generation:** Use a large language model to generate summaries, answer questions, or extract key findings by combining retrieved information.

### 3. Visualization & Dashboard

#### Dashboard Design:

- Use **Streamlit** for creating interactive visualizations.

#### User Interface:

- Develop an interface where researchers can:
  - Enter a query or set of keywords.
  - View summaries and synthesized insights.

## Directory Structure

```plaintext
ai-research-assistant
├── src
│   ├── data
│   │   ├── __init__.py
│   │   └── data_fetch.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── text_processing.py
│   │   └── vector_store.py
│   ├── synthesizer
│   │   ├── __init__.py
│   │   └── rag_pipeline.py
│   ├── utils
│   │   └── __init__.py
│   ├── summarizer
│   │   └── __init__.py
│   ├── dashboard.py
│   └── main.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

Clone the repository:

```sh
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant
```

Create a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate
```

Install the required libraries:

```sh
pip install -r requirements.txt
```

## Usage

### Running the Data Fetch Script

To fetch data from the arXiv API, run the following command:

```sh
python src/data/data_fetch.py
```

### Running the RAG Pipeline

To run the RAG pipeline, execute the following command:

```sh
export PYTHONPATH=$(pwd)
python src/synthesizer/rag_pipeline.py
```

### Running the Streamlit Dashboard

To start the Streamlit dashboard, execute the following command:

```sh
export PYTHONPATH=$(pwd)
streamlit run src/dashboard.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

- arXiv API
- Sentence Transformers
- FAISS
- Streamlit

