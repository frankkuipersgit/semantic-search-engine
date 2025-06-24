import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import requests
import json
import concurrent.futures
import os
import hashlib
import pymongo
import openai
import tiktoken
import dotenv
from datetime import datetime

dotenv.load_dotenv()

def stqdm(iterable, desc=None):
    from time import sleep
    total = len(iterable)
    bar = st.progress(0)
    for i, item in enumerate(iterable):
        yield item
        bar.progress((i+1)/total)
        sleep(0.01)
    bar.empty()

# --- CONFIG ---
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
    device_label = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    DEVICE = "cuda"
    device_label = "CUDA GPU"
else:
    DEVICE = "cpu"
    device_label = "CPU"

st.title("🔎 Semantic Search Engine (Dutch & Multilingual)")
st.write(f"**Embedding device:** {device_label}")

# Backend selector
backend = "MongoDB + OpenAI"

# --- Hardcoded MongoDB + OpenAI config ---
mongo_uri = os.environ.get("MONGO_URI")
mongo_db_name = os.environ.get("MONGO_DB_NAME")
mongo_collection_name = os.environ.get("MONGO_COLLECTION_NAME")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# OpenAI embedding model switcher
openai_model_options = ["text-embedding-3-large"]
OPENAI_EMBEDDING_MODEL = st.selectbox("OpenAI embedding model", openai_model_options, index=0)

# Database clear button (only for MongoDB + OpenAI backend)
st.sidebar.write("---")
confirm_clear = st.sidebar.checkbox("Bevestig: database legen")
if st.sidebar.button("Leeg database (MongoDB)"):
    if confirm_clear:
        client = pymongo.MongoClient(mongo_uri)
        db = client[mongo_db_name]
        collection = db[mongo_collection_name]
        deleted = collection.delete_many({})
        st.sidebar.success(f"Database geleegd! ({deleted.deleted_count} documenten verwijderd)")
    else:
        st.sidebar.warning("Vink eerst de bevestiging aan voordat je de database leegt.")

# Session state for caching embeddings and selections
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'text_fields' not in st.session_state:
    st.session_state.text_fields = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'building' not in st.session_state:
    st.session_state.building = False
if 'built_for' not in st.session_state:
    st.session_state.built_for = {'df_hash': None, 'fields': None, 'batch_size': None, 'backend': None}
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 128

# Paths for local cache
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "index_metadata.json"

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file).fillna("")
    
    # Sanitize string columns to remove null bytes which are invalid for BSON
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace('\x00', '', regex=False)

    st.write("Data preview:", df.head())

    # Image column selector
    image_col = st.selectbox("Selecteer een kolom met afbeeldingen (optioneel)", options=["(geen)"] + list(df.columns), index=0)

    # Field selector
    text_fields = st.multiselect(
        "Select fields to use for search",
        options=list(df.columns),
        default=list(df.columns)[:2] if len(df.columns) >= 2 else list(df.columns)
    )

    # Batch size slider (only for local)
    batch_size = None

    # Helper function to compute a unique hash for the current data/fields/backend
    def compute_cache_hash(df, text_fields, batch_size, backend):
        df_hash = pd.util.hash_pandas_object(df).sum()
        fields_hash = hash(tuple(text_fields))
        return hashlib.md5(f"{df_hash}_{fields_hash}_{batch_size}_{backend}".encode()).hexdigest()

    # Hash for current data/fields/batch_size/backend to check if embeddings are up to date
    df_hash = pd.util.hash_pandas_object(df).sum()
    fields_hash = tuple(text_fields)
    cache_hash = compute_cache_hash(df, text_fields, batch_size, backend)

    # Try to load from cache (MongoDB metadata only)
    cache_loaded = False
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
        if meta.get("cache_hash") == cache_hash and meta.get("backend") == backend:
            st.session_state.embeddings = np.array([])  # Not used for Mongo
            st.session_state.index = None
            st.session_state.model = None
            st.session_state.built_for = {'df_hash': df_hash, 'fields': fields_hash, 'batch_size': batch_size, 'backend': backend}
            cache_loaded = True
            st.success("MongoDB + OpenAI embeddings/index already processed for this data/fields.")

    # If new file or new fields or batch size or backend, reset embeddings (but don't auto-build)
    if not cache_loaded and (
        st.session_state.df is None or
        not st.session_state.df.equals(df) or
        st.session_state.text_fields != text_fields or
        st.session_state.built_for.get('batch_size') != batch_size or
        st.session_state.built_for.get('backend') != backend
    ):
        st.session_state.df = df.copy()
        st.session_state.text_fields = text_fields
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.model = None
        st.session_state.built_for = {'df_hash': None, 'fields': None, 'batch_size': None, 'backend': None}

    if text_fields:
        # Combine selected fields
        df["combined_text"] = df[text_fields].astype(str).agg(" ".join, axis=1)

        # Build Embeddings Button
        build_btn = st.button(
            "Build Embeddings",
            disabled=st.session_state.building or (not text_fields) or (not mongo_uri or not openai_api_key or not mongo_db_name or not mongo_collection_name)
        )

        # Build embeddings only when button is pressed
        if build_btn and not st.session_state.building and not cache_loaded:
            st.toast("Building embeddings, please wait...")
            st.session_state.building = True
            st.session_state.embeddings = None
            st.session_state.index = None
            st.session_state.model = None
            with st.spinner("Creating embeddings with OpenAI and uploading to MongoDB..."):
                openai.api_key = openai_api_key
                client = pymongo.MongoClient(mongo_uri)
                db = client[mongo_db_name]
                collection = db[mongo_collection_name]
                # Remove all previous docs for this session (optional)
                collection.delete_many({"_session": cache_hash})
                docs = [t[:2000] for t in df["combined_text"].tolist()]  # Truncate to 2000 chars for OpenAI

                # Dynamisch batchen op tokenlimiet
                enc = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
                max_tokens_per_request = 300000
                batches = []
                current_batch = []
                current_tokens = 0
                for text in docs:
                    tokens = len(enc.encode(text))
                    if current_tokens + tokens > max_tokens_per_request and current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_tokens = 0
                    current_batch.append(text)
                    current_tokens += tokens
                if current_batch:
                    batches.append(current_batch)

                max_workers = 4
                all_embeddings = []
                st.toast("Starting embedding and upload to MongoDB (parallel batches)...")
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                total_batches = len(batches)

                def embed_batch(batch):
                    return openai.embeddings.create(
                        input=batch,
                        model=OPENAI_EMBEDDING_MODEL
                    )

                responses = [None] * len(batches)
                from concurrent.futures import ThreadPoolExecutor, as_completed
                done_batches = 0
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {executor.submit(embed_batch, batch): idx for idx, batch in enumerate(batches)}
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            responses[idx] = future.result()
                        except Exception as e:
                            st.error(f"Fout bij batch {idx+1}: {e}")
                        done_batches += 1
                        progress = done_batches / total_batches
                        progress_bar.progress(progress)
                        status_placeholder.markdown(f"**Batch {done_batches} van {total_batches} verwerkt**")
                        st.toast(f"Batch {done_batches} van {total_batches} verwerkt")
                progress_bar.empty()
                status_placeholder.empty()

                # Verzamel en upload alle embeddings naar MongoDB
                doc_idx = 0
                for batch_idx, response in enumerate(responses):
                    if response is None:
                        continue
                    for j, emb in enumerate(response.data):
                        all_embeddings.append(emb.embedding)
                        doc = {
                            "_session": cache_hash,
                            "row": doc_idx,
                            # Truncate text to 2000 chars and ensure string type
                            "text": str(docs[doc_idx])[:2000],
                            "embedding": emb.embedding,
                            # Add all original fields for reference
                        }
                        for col in df.columns:
                            val = df.iloc[doc_idx][col]
                            # Convert numpy scalars to native Python types
                            if hasattr(val, 'item'):
                                try:
                                    val = val.item()
                                except Exception:
                                    val = str(val)
                            else:
                                val = str(val)
                            # Truncate if string
                            if isinstance(val, str):
                                val = val[:2000]
                            doc[col] = val
                        # Save image column if selected
                        if image_col != "(geen)":
                            doc["image"] = str(df.iloc[doc_idx][image_col])
                        # Only insert if SKU+_session does not exist
                        sku_val = doc.get('sku', None)
                        if sku_val is not None:
                            exists = collection.count_documents({"_session": cache_hash, "sku": sku_val}, limit=1)
                            if exists == 0:
                                collection.insert_one(doc)
                        else:
                            # If no SKU, fallback to row+_session
                            exists = collection.count_documents({"_session": cache_hash, "row": doc["row"]}, limit=1)
                            if exists == 0:
                                collection.insert_one(doc)
                        doc_idx += 1
                st.toast("All embeddings created and uploaded to MongoDB!")
                st.session_state.embeddings = np.array(all_embeddings, dtype=np.float32)
                st.session_state.index = None
                st.session_state.model = None
                st.session_state.built_for = {'df_hash': df_hash, 'fields': fields_hash, 'batch_size': batch_size, 'backend': backend}
                # Save metadata to disk
                with open(METADATA_PATH, "w") as f:
                    json.dump({"cache_hash": cache_hash, "backend": backend}, f)
            st.success("Embeddings created and uploaded to MongoDB! Ready to search. (Metadata saved)")
            st.session_state.building = False

        # Search UI
        query = st.text_input("Enter your search query")
        k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        if not cache_loaded:
            st.info("Please build embeddings before searching.")
        elif query:
            with st.spinner("Embedding query and searching MongoDB..."):
                openai.api_key = openai_api_key
                client = pymongo.MongoClient(mongo_uri)
                db = client[mongo_db_name]
                collection = db[mongo_collection_name]
                # Embed the query
                response = openai.embeddings.create(
                    input=[query],
                    model=OPENAI_EMBEDDING_MODEL
                )
                query_vec = response.data[0].embedding
                # Run $vectorSearch aggregation
                pipeline = [
                    {"$vectorSearch": {
                        "index": "vector_index",  # You must create this index in Atlas UI
                        "path": "embedding",
                        "queryVector": query_vec,
                        "numCandidates": max(100, k*20),
                        "limit": k,
                        "filter": {"_session": cache_hash}
                    }},
                    {"$project": {
                        "_id": 0,
                        "row": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        **{col: 1 for col in df.columns}
                    }}
                ]
                results = list(collection.aggregate(pipeline))
                st.subheader("Top matches:")
                for rank, item in enumerate(results):
                    idx = item.get("row", None)
                    score = item.get("score", 0)
                    cols = st.columns([1, 2, 6])
                    if 'sku' in item:
                        cols[0].write(f"SKU: {item['sku']}")
                    if 'image' in item:
                        cols[1].image(item['image'], width=100)
                    for col in text_fields:
                        if col in item:
                            cols[2].write(f"{col.capitalize()}: {item[col]}")
                    st.write(f"Score: {score:.3f}")
                    st.write("---")
    else:
        st.warning("Please select at least one field to search through.")
else:
    if backend == "MongoDB + OpenAI":
        # Try to fetch a sample document to infer fields
        openai.api_key = openai_api_key
        client = pymongo.MongoClient(mongo_uri)
        db = client[mongo_db_name]
        collection = db[mongo_collection_name]
        sample_doc = collection.find_one()
        if sample_doc:
            # Exclude internal fields
            default_fields = [k for k in sample_doc.keys() if k not in ["_id", "embedding", "_session", "row", "score"]]
        else:
            default_fields = []
        text_fields = st.multiselect(
            "Select fields to display in results",
            options=default_fields,
            default=default_fields[:2] if len(default_fields) >= 2 else default_fields
        )
        query = st.text_input("Enter your search query")
        k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        if not sample_doc:
            st.info("No documents found in MongoDB. Please upload a CSV and build embeddings first.")
        elif query:
            with st.spinner("Embedding query and searching MongoDB..."):
                response = openai.embeddings.create(
                    input=[query],
                    model=OPENAI_EMBEDDING_MODEL
                )
                query_vec = response.data[0].embedding
                pipeline = [
                    {"$vectorSearch": {
                        "index": "vector_index",  # Use correct index name
                        "path": "embedding",
                        "queryVector": query_vec,
                        "numCandidates": max(100, k*20),
                        "limit": k
                    }},
                    {"$project": {
                        "_id": 0,
                        "row": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        **{col: 1 for col in default_fields}
                    }}
                ]
                results = list(collection.aggregate(pipeline))
                st.subheader("Top matches:")
                for rank, item in enumerate(results):
                    idx = item.get("row", None)
                    score = item.get("score", 0)
                    cols = st.columns([1, 2, 6])
                    if 'sku' in item:
                        cols[0].write(f"SKU: {item['sku']}")
                    if 'image' in item:
                        cols[1].image(item['image'], width=100)
                    for col in text_fields:
                        if col in item:
                            cols[2].write(f"{col.capitalize()}: {item[col]}")
                    st.write(f"Score: {score:.3f}")
                    st.write("---")
    else:
        st.info("Please upload a CSV file to get started.")

# MongoDB connection status
try:
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    client.server_info()
    mongo_status = "🟢 Verbonden"
    db = client[mongo_db_name]
    collection = db[mongo_collection_name]
    doc_count = collection.count_documents({})
    st.sidebar.markdown(f"**DB:** `{mongo_db_name}`  ")
    st.sidebar.markdown(f"**Collectie:** `{mongo_collection_name}`  ")
    st.sidebar.markdown(f"**Docs:** `{doc_count}`  ")
except Exception as e:
    mongo_status = f"🔴 Niet verbonden: {e}"
    st.sidebar.markdown(f"**DB:** `{mongo_db_name}`  ")
    st.sidebar.markdown(f"**Collectie:** `{mongo_collection_name}`  ")
    st.sidebar.markdown(f"**Docs:** `?`  ")
st.sidebar.markdown(f"**MongoDB status:** {mongo_status}")
st.sidebar.markdown(f"**Model:** `{OPENAI_EMBEDDING_MODEL}`")

# Tabs for workflow
main_tabs = st.tabs(["Upload & Index", "Search Products", "Search Orders", "Categorize Products", "Dataset Info"])

# --- TAB 1: Upload & Index ---
with main_tabs[0]:
    # ... existing upload & index code ...
    pass

# --- TAB 2: Search Products ---
with main_tabs[1]:
    st.header("2. Search Products")
    # ... existing search products code ...
    pass

# --- TAB 3: Search Orders (disabled/coming soon) ---
with main_tabs[2]:
    st.header("3. Search Orders")
    st.info("Deze functionaliteit komt binnenkort beschikbaar.")
    st.markdown("<div style='color:gray;'>[Gedisabled]</div>", unsafe_allow_html=True)

# --- TAB 4: Categorize Products (disabled/coming soon) ---
with main_tabs[3]:
    st.header("4. Categorize Products")
    st.info("Deze functionaliteit komt binnenkort beschikbaar.")
    st.markdown("<div style='color:gray;'>[Gedisabled]</div>", unsafe_allow_html=True)

# --- TAB 5: Dataset Info ---
with main_tabs[4]:
    st.header("5. Dataset Info & Statistieken")
    pass
