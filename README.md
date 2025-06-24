# Semantic Search Engine

This project is a Streamlit-based semantic search engine using OpenAI embeddings and MongoDB Atlas Vector Search.

## Features

- Upload CSV files and build semantic search indexes using OpenAI embeddings
- Store and search embeddings in MongoDB Atlas
- Parallel and batch processing for fast embedding
- Image column support for search results
- Secure secret management with `.env` file

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd semantic-search-engine
```

### 2. Create a `.env` file

Create a file named `.env` in the project root with the following content (replace with your own credentials):

```
MONGO_URI=your_mongodb_uri
MONGO_DB_NAME=vector
MONGO_COLLECTION_NAME=documents
OPENAI_API_KEY=your_openai_api_key
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run search.py
```

## Notes

- The `.env` file is ignored by git (see `.gitignore`).
- All secrets and private keys are loaded from `.env` using `python-dotenv`.
- Make sure your MongoDB Atlas vector index is configured with the correct dimensions and filter fields (see code and Atlas docs).

## Requirements

- Python 3.8+
- OpenAI API key
- MongoDB Atlas cluster with vector search enabled

## License

MIT

---

Feel free to modify or extend the app as needed!
