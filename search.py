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
from pymongo import UpdateOne

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

# --- (Moved UI title and device label below, inside tab logic) ---

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
        # Use the selected collection from session state, fallback to default
        collection_name_to_clear = st.session_state.mongo_collection_name if 'mongo_collection_name' in st.session_state else mongo_collection_name
        collection = db[collection_name_to_clear]
        deleted = collection.delete_many({})
        st.sidebar.success(f"Database geleegd! ({deleted.deleted_count} documenten verwijderd uit collectie '{collection_name_to_clear}')")
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

    # Identifier column selector
    identifier_col = st.selectbox("Selecteer een kolom als unieke identifier (bijv. SKU, product ID)", options=["(geen)"] + list(df.columns), index=0)
    st.session_state.identifier_col = identifier_col

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
            # Check MongoDB for documents with this _session
            try:
                client = pymongo.MongoClient(mongo_uri)
                db = client[mongo_db_name]
                collection = db[mongo_collection_name]
                doc_count = collection.count_documents({"_session": cache_hash})
            except Exception as e:
                doc_count = 0
            if doc_count > 0:
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
            # Deduplication: Only embed products not already in the DB by identifier
            if identifier_col == "(geen)":
                st.warning("Selecteer een kolom als unieke identifier voordat je embeddings bouwt.")
            else:
                st.toast("Building embeddings, please wait...")
                st.session_state.building = True
                st.session_state.embeddings = None
                st.session_state.index = None
                st.session_state.model = None
                with st.spinner("Creating embeddings with OpenAI and uploading to MongoDB..."):
                    openai.api_key = openai_api_key
                    client = pymongo.MongoClient(mongo_uri)
                    db = client[mongo_db_name]
                    # Use the selected collection from session state
                    collection = db[st.session_state.mongo_collection_name] if 'mongo_collection_name' in st.session_state else db[mongo_collection_name]
                    # Query all existing identifiers
                    existing_ids = set()
                    for doc in collection.find({}, {identifier_col: 1}):
                        val = doc.get(identifier_col)
                        if val is not None:
                            existing_ids.add(str(val))
                    # Only keep rows whose identifier is not in existing_ids
                    mask = ~df[identifier_col].astype(str).isin(existing_ids)
                    df_new = df[mask].copy()
                    if df_new.empty:
                        st.success("Alle producten in deze CSV zijn al aanwezig in de database. Geen nieuwe producten om te embedden.")
                        st.session_state.building = False
                    else:
                        # Combine selected fields for new products
                        df_new["combined_text"] = df_new[text_fields].astype(str).agg(" ".join, axis=1)
                        # Remove all previous docs for this session (optional, or skip for dedup)
                        # collection.delete_many({"_session": cache_hash})
                        docs = [str(t)[:2000] for t in df_new["combined_text"].tolist() if isinstance(t, str) and str(t).strip()]
                        MAX_BATCH_SIZE = 512
                        MAX_CHAR_LENGTH = 16384  # ~2x max tokens for safety
                        enc = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
                        max_tokens_per_request = 300000
                        batches = []
                        current_batch = []
                        current_tokens = 0
                        for text in docs:
                            if not isinstance(text, str) or not text.strip():
                                continue  # skip empty or non-string
                            text = text[:MAX_CHAR_LENGTH]  # truncate overly long strings
                            tokens = len(enc.encode(text))
                            if ((current_tokens + tokens > max_tokens_per_request) or (len(current_batch) >= MAX_BATCH_SIZE)) and current_batch:
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
                        import time
                        def embed_batch(batch, batch_idx):
                            clean_batch = [str(x)[:MAX_CHAR_LENGTH] for x in batch if isinstance(x, str) and str(x).strip()]
                            if not clean_batch:
                                raise ValueError(f"Batch {batch_idx} is empty or contains only invalid strings.")
                            print(f"Embedding batch {batch_idx} of size {len(clean_batch)}. Sample: {clean_batch[:2]}")
                            retries = 3
                            while retries > 0:
                                try:
                                    return openai.embeddings.create(
                                        input=clean_batch,
                                        model=OPENAI_EMBEDDING_MODEL
                                    )
                                except Exception as e:
                                    print(f"Rate limit error on batch {batch_idx}: {e}. Retrying after 12s...")
                                    time.sleep(12)
                                    retries -= 1
                            return None
                        responses = [None] * len(batches)
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        done_batches = 0
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_idx = {executor.submit(embed_batch, batch, idx): idx for idx, batch in enumerate(batches)}
                            for future in as_completed(future_to_idx):
                                idx = future_to_idx[future]
                                try:
                                    response = future.result()
                                    if response is None:
                                        st.error(f"Fout bij batch {idx+1}: batch overgeslagen wegens input error.")
                                        continue
                                    docs_to_upload = []
                                    start_idx = sum(len(b) for b in batches[:idx])
                                    for j, emb in enumerate(response.data):
                                        doc_idx = df_new.index[start_idx + j]
                                        doc = {
                                            "_session": cache_hash,
                                            "row": int(doc_idx),
                                            "text": str(df_new.loc[doc_idx, "combined_text"])[:2000],
                                            "embedding": emb.embedding,
                                        }
                                        for col in df_new.columns:
                                            val = df_new.loc[doc_idx, col]
                                            if hasattr(val, 'item'):
                                                try:
                                                    val = val.item()
                                                except Exception:
                                                    val = str(val)
                                            else:
                                                val = str(val)
                                            if isinstance(val, str):
                                                val = val[:2000]
                                            doc[col] = val
                                        if image_col != "(geen)":
                                            doc["image"] = str(df_new.loc[doc_idx, image_col])
                                        docs_to_upload.append(doc)
                                    if docs_to_upload:
                                        collection.insert_many(docs_to_upload)
                                except Exception as e:
                                    st.error(f"Fout bij batch {idx+1}: {e}")
                                done_batches += 1
                                progress = done_batches / total_batches
                                progress_bar.progress(progress)
                                status_placeholder.markdown(f"**Batch {done_batches} van {total_batches} (geüpload)**")
                                st.toast(f"Batch {done_batches} van {total_batches} (geüpload)")
                        progress_bar.empty()
                        status_placeholder.empty()
                        st.toast("All embeddings created and uploaded to MongoDB!")
                        st.session_state.index = None
                        st.session_state.model = None
                        st.session_state.built_for = {'df_hash': df_hash, 'fields': fields_hash, 'batch_size': batch_size, 'backend': backend}
                        with open(METADATA_PATH, "w") as f:
                            json.dump({"cache_hash": cache_hash, "backend": backend}, f)
                    st.success("Embeddings created and uploaded to MongoDB! Ready to search. (Metadata saved)")
                    st.session_state.building = False

# Ensure MongoDB client is defined for all tabs
client = pymongo.MongoClient("mongodb://frank:adfgaSDRSDGsdg5rthdts5ey6dr5rxtdyjftt@116.203.123.107:27017/")

# Sidebar navigation
sidebar_tabs = [
    "Search Products",
    "Search Orders",
    "Categorize Products",
    "Dataset Info",
    "Upload & Index"
]
selected_tab = st.sidebar.radio("Navigatie", sidebar_tabs, index=0)

# --- MAIN AREA ---
if selected_tab == "Upload & Index":
    st.header("Upload & Indexeer je CSV")
    pass
elif selected_tab == "Search Products":
    st.title("🔎 Semantic Search Engine (Dutch & Multilingual)")
    st.write(f"**Embedding device:** {device_label}")
    st.header("Search Products")
    db = client[mongo_db_name]
    collection = db[mongo_collection_name]
    sample_doc = collection.find_one()
    if sample_doc:
        default_fields = [k for k in sample_doc.keys() if k not in ["_id", "embedding", "_session", "row", "score"]]
    else:
        default_fields = []
    # --- Dynamic Filters ---
    filter_dict = {}
    st.subheader("Filters")
    if sample_doc:
        for field in default_fields:
            val = sample_doc[field]
            if isinstance(val, (int, float, np.integer, np.floating)):
                min_val = st.number_input(f"{field} min", value=None, step=1.0, format="%.2f")
                max_val = st.number_input(f"{field} max", value=None, step=1.0, format="%.2f")
                if min_val is not None:
                    filter_dict[field] = filter_dict.get(field, {})
                    filter_dict[field]["$gte"] = min_val
                if max_val is not None:
                    filter_dict[field] = filter_dict.get(field, {})
                    filter_dict[field]["$lte"] = max_val
                if field in filter_dict and not filter_dict[field]:
                    filter_dict.pop(field)
            elif isinstance(val, bool):
                bool_val = st.selectbox(f"{field}", options=["(geen filter)", True, False], index=0)
                if bool_val != "(geen filter)":
                    filter_dict[field] = bool_val
            elif isinstance(val, str):
                text_filter = st.text_input(f"{field} bevat", value="")
                if text_filter:
                    filter_dict[field] = {"$regex": text_filter, "$options": "i"}
    text_fields = st.multiselect(
        "Velden om te tonen in zoekresultaten",
        options=default_fields,
        default=default_fields[:2] if len(default_fields) >= 2 else default_fields,
        help="Kies welke velden je wilt zien in de zoekresultaten."
    )
    query = st.text_input("Voer je zoekopdracht in")
    k = st.slider("Aantal resultaten", min_value=1, max_value=10, value=5)
    if not sample_doc:
        st.info("Geen documenten gevonden in MongoDB. Upload en indexeer eerst een CSV.")
    elif query:
        st.info(f"Zoekt met model: `{OPENAI_EMBEDDING_MODEL}`")
        with st.spinner("Query embedden en zoeken in MongoDB..."):
            openai.api_key = openai_api_key
            response = openai.embeddings.create(
                input=[query],
                model=OPENAI_EMBEDDING_MODEL
            )
            query_vec = response.data[0].embedding
            # Always filter on _session if present in sample_doc
            session_filter = {}
            if "_session" in sample_doc:
                session_filter["_session"] = sample_doc["_session"]
            # Merge with UI filters
            mongo_filter = {**session_filter, **filter_dict}
            pipeline = [
                {"$vectorSearch": {
                    "index": "vector_index",
                    "path": "em.edding",
                    "queryVector": query_vec,
                    "numCandidates": max(100, k*20),
                    "limit": k,
                    "filter": mongo_filter if mongo_filter else None
                }},
                {"$project": {
                    "_id": 0,
                    "row": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    **{col: 1 for col in default_fields}
                }}
            ]
            # Remove None filter if empty
            if pipeline[0]["$vectorSearch"]["filter"] is None:
                del pipeline[0]["$vectorSearch"]["filter"]
            results = list(collection.aggregate(pipeline))
            st.subheader("Top matches:")
            if not results:
                st.warning("Geen resultaten gevonden.")
            for rank, item in enumerate(results):
                score = item.get("score", 0)
                with st.container():
                    cols = st.columns([1, 2, 6])
                    if 'sku' in item:
                        cols[0].write(f"SKU: {item['sku']}")
                    if 'image' in item:
                        cols[1].image(item['image'], width=100)
                    for col in text_fields:
                        if col in item:
                            cols[2].write(f"{col.capitalize()}: {item[col]}")
                    cols[2].markdown(f"**Score:** `{score:.3f}`")
                    with st.expander("Toon raw document"):
                        st.json(item)
                st.markdown("---")
elif selected_tab == "Search Orders":
    st.header("Search Orders")
    st.info("Deze functionaliteit komt binnenkort beschikbaar.")
    st.markdown("<div style='color:gray;'>[Gedisabled]</div>", unsafe_allow_html=True)
elif selected_tab == "Categorize Products":
    st.header("Categorize Products")
    db = client[mongo_db_name]
    collection = db["documents"]
    cat_collection = db["categories"]

    # Editable table for categories
    st.subheader("Stap 1: Voeg categorieën toe of bewerk ze")
    existing_cats = list(cat_collection.find({}, {"_id": 0, "category": 1}))
    cat_df = pd.DataFrame(existing_cats) if existing_cats else pd.DataFrame({"category": [""]})
    cat_df = st.data_editor(cat_df, num_rows="dynamic", key="cat_editor")
    if st.button("Sla categorieën op"):
        categories_to_save = [row["category"] for _, row in cat_df.iterrows() if row["category"].strip()]
        if not categories_to_save:
            st.warning("Geen categorieën om op te slaan.")
        else:
            with st.spinner("Categorieën embedden en opslaan..."):
                openai.api_key = openai_api_key
                try:
                    cat_embeddings_response = openai.embeddings.create(
                        input=categories_to_save,
                        model=OPENAI_EMBEDDING_MODEL
                    )
                    docs_to_insert = []
                    for i, category_name in enumerate(categories_to_save):
                        docs_to_insert.append({
                            "category": category_name,
                            "embedding": cat_embeddings_response.data[i].embedding
                        })
                    cat_collection.delete_many({})
                    if docs_to_insert:
                        cat_collection.insert_many(docs_to_insert)
                    st.success(f"{len(docs_to_insert)} categorieën opgeslagen met embeddings!")
                except Exception as e:
                    st.error(f"Fout bij het embedden van categorieën: {e}")

    # Categorize products
    st.subheader("Stap 2: Categoriseer producten met AI")
    if st.button("Categoriseer producten"):
        categories_with_embeddings = list(cat_collection.find({}, {"_id": 0, "category": 1, "embedding": 1}))
        if not categories_with_embeddings:
            st.warning("Voeg eerst categorieën toe en sla ze op.")
        else:
            categories = [c["category"] for c in categories_with_embeddings]
            cat_vecs = np.array([c["embedding"] for c in categories_with_embeddings], dtype=np.float32)
            
            openai.api_key = openai_api_key
            
            total_products = collection.count_documents({})
            if total_products == 0:
                st.warning("Geen producten gevonden om te categoriseren. Indexeer eerst een CSV.")
                st.stop()

            # --- Parallel processing with streaming updates ---
            chunk_size = 100  # Smaller chunks for more frequent updates
            batch_size = 10   # Process multiple chunks in parallel
            
            # Get all product IDs first for better progress tracking
            product_ids = list(collection.find({}, {"_id": 1}))
            total_products = len(product_ids)
            
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            table_placeholder = st.empty()
            
            st.toast(f"Categorizing {total_products} products in parallel chunks...")
            
            # Process chunks in parallel
            def process_chunk(chunk_ids):
                """Process a chunk of products and return results"""
                chunk_products = list(collection.find({"_id": {"$in": chunk_ids}}))
                if not chunk_products or "embedding" not in chunk_products[0]:
                    return []
                
                prod_vecs = np.array([p["embedding"] for p in chunk_products], dtype=np.float32)
                sim = np.dot(prod_vecs, cat_vecs.T)
                best_cat_idx = np.argmax(sim, axis=1)
                best_cat_score = np.max(sim, axis=1)
                
                # Prepare results and updates
                results = []
                update_operations = []
                
                for j, p_item in enumerate(chunk_products):
                    p_item["assigned_category"] = categories[best_cat_idx[j]]
                    p_item["category_score"] = float(best_cat_score[j])
                    results.append(p_item)
                    update_operations.append(
                        UpdateOne({"_id": p_item["_id"]}, {"$set": {
                            "assigned_category": p_item["assigned_category"], 
                            "category_score": p_item["category_score"]
                        }})
                    )
                
                # Bulk update this chunk
                if update_operations:
                    collection.bulk_write(update_operations)
                
                return results
            
            # Split product IDs into chunks
            chunks = [product_ids[i:i + chunk_size] for i in range(0, len(product_ids), chunk_size)]
            chunk_ids = [chunk for chunk in chunks]
            
            # Process chunks in parallel with ThreadPoolExecutor
            all_processed_products = []
            display_columns = None
            processed_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all chunks for processing
                future_to_chunk = {executor.submit(process_chunk, [p["_id"] for p in chunk]): chunk for chunk in chunk_ids}
                
                # Process completed chunks as they finish
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        all_processed_products.extend(chunk_results)
                        processed_count += len(chunk_results)
                        
                        # Update progress
                        progress = processed_count / total_products
                        progress_bar.progress(progress)
                        status_placeholder.markdown(f"**Verwerkt:** {processed_count}/{total_products} producten ({progress:.1%})")
                        
                        # Update table every few chunks for better performance
                        if len(all_processed_products) > 0 and (processed_count % (chunk_size * 5) == 0 or processed_count == total_products):
                            if display_columns is None:
                                # Determine columns on first run
                                preferred_order = ["sku", "text", "assigned_category", "category_score"]
                                all_keys = list(all_processed_products[0].keys())
                                other_cols = sorted([k for k in all_keys if k not in preferred_order and k not in ["_id", "embedding", "row", "_session"]])
                                display_columns = [col for col in preferred_order if col in all_keys] + other_cols
                            
                            # Show latest results
                            df_stream = pd.DataFrame(all_processed_products[-500:])  # Show last 500 for performance
                            if 'category_score' in df_stream.columns:
                                df_stream['category_score'] = df_stream['category_score'].apply(lambda x: f'{x:.4f}')
                            table_placeholder.dataframe(df_stream.iloc[::-1][display_columns])

                    except Exception as e:
                        st.error(f"Fout bij verwerken van chunk: {e}")
            
            st.toast("Categorization complete!")
            progress_bar.empty()
            status_placeholder.empty()

            # --- Add Graphs ---
            st.subheader("Analyse van de categorisatie")
            final_df = pd.DataFrame(all_processed_products)

            if not final_df.empty:
                # 1. Histogram of scores
                st.write("##### Verdeling van de scores")
                hist_values = np.histogram(
                    final_df['category_score'], bins=20, range=(0,1))[0]
                st.bar_chart(hist_values)

                # 2. Products per category
                st.write("##### Aantal producten per categorie")
                category_counts = final_df['assigned_category'].value_counts()
                st.bar_chart(category_counts)
            else:
                st.warning("Geen data beschikbaar voor analyse.")

    st.info("Voeg eerst categorieën toe, klik dan op 'Categoriseer producten'. Je kunt het resultaat downloaden als CSV.")
elif selected_tab == "Dataset Info":
    st.header("Dataset Info & Statistieken")
    pass

# --- SIDEBAR DB INFO ---
with st.sidebar.expander("📦 Database Info & Config", expanded=True):
    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        client.server_info()
        mongo_status = "🟢 Verbonden"
        db = client[mongo_db_name]
        # List all collections in the database
        collection_names = db.list_collection_names()
        # Use session state or default
        if 'mongo_collection_name' not in st.session_state:
            st.session_state.mongo_collection_name = mongo_collection_name if mongo_collection_name in collection_names else (collection_names[0] if collection_names else "")
        selected_collection = st.selectbox("Selecteer collectie", options=collection_names, index=collection_names.index(st.session_state.mongo_collection_name) if st.session_state.mongo_collection_name in collection_names else 0)
        st.session_state.mongo_collection_name = selected_collection
        collection = db[selected_collection]
        doc_count = collection.count_documents({})
        st.markdown(f"**DB:** `{mongo_db_name}`  ")
        st.markdown(f"**Collectie:** `{selected_collection}`  ")
        st.markdown(f"**Docs:** `{doc_count}`  ")
    except Exception as e:
        mongo_status = f"🔴 Niet verbonden: {e}"
        st.markdown(f"**DB:** `{mongo_db_name}`  ")
        st.markdown(f"**Collectie:** `{mongo_collection_name}`  ")
        st.markdown(f"**Docs:** `?`  ")
    st.markdown(f"**MongoDB status:** {mongo_status}")
    st.markdown(f"**Model:** `{OPENAI_EMBEDDING_MODEL}`")
