import streamlit as st
import sys
import sqlite3

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
import pymongo
import os
import openai
import pandas as pd
import json
import ssl
import datetime

# Initialize OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002"
)

# Initialize MongoDB with better error handling and DNS configuration
def get_mongo_client():
    try:
        # Configure MongoDB client for Atlas cluster
        client = pymongo.MongoClient(
            st.secrets["MONGODB_URI"],
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            retryWrites=True,
            w='majority'
        )
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"MongoDB Connection Error: {str(e)}")
        return None

# Initialize global variables
mongo_client = get_mongo_client()
if mongo_client:
    db = mongo_client.legal_documents
else:
    st.error("Failed to initialize MongoDB connection. Some features may not work.")

def analyze_intent(query):
    system_prompt = """You are an AI that analyzes user queries about legal documents. 
    Determine if the user wants to process:
    1. Only MSA documents
    2. Only NDA documents
    3. Both MSA and NDA documents
    
    Respond with exactly one of these strings: "MSA", "NDA", or "BOTH"."""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message['content'].strip()

def extract_key_terms(text, doc_type):
    system_prompt = ""
    if doc_type == "MSA":
        system_prompt = """You are a legal expert specializing in Master Service Agreements. Extract key terms and return them in the following strict JSON format:
        {
            "parties_involved": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ],
            "effective_date": {"term": "exact date from document", "description": "date context"},
            "service_scope": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ],
            "payment_terms": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ],
            "service_levels": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ],
            "liability_terms": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ]
        }"""
    else:  # NDA
        system_prompt = """You are a legal expert specializing in NDAs. Extract key terms and return them in the following strict JSON format:
        {
            "parties_involved": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ],
            "effective_date": {"term": "exact date from document", "description": "date context"},
            "confidential_info": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ],
            "permitted_use": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ],
            "protection_measures": [
                {"term": "exact quote from document", "description": "detailed explanation"}
            ]
        }"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        content = response.choices[0].message['content'].strip()
        
        # Try to find JSON content if there's any extra text
        if not content.startswith('{'):
            import re
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                content = json_match.group(1)
        
        # Parse and validate JSON
        extracted_data = json.loads(content)
        
        # Ensure minimum structure
        required_fields = ['parties_involved', 'effective_date']
        if not all(field in extracted_data for field in required_fields):
            raise ValueError("Missing required fields in extraction")
        
        return extracted_data
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        print("Raw response:", response.choices[0].message.content)
        # Return a minimal valid structure instead of None
        return {
            "parties_involved": [{"term": "Not found", "description": "Could not extract"}],
            "effective_date": {"term": "Not found", "description": "Could not extract"}
        }
    except Exception as e:
        st.error(f"Error in key terms extraction: {str(e)}")
        # Return minimal valid structure
        return {
            "parties_involved": [{"term": "Not found", "description": "Could not extract"}],
            "effective_date": {"term": "Not found", "description": "Could not extract"}
        }

def search_documents(query, doc_type):
    results = []
    try:
        if doc_type == "MSA" or doc_type == "BOTH":
            msa_collection = client.get_collection("msa_documents", embedding_function=openai_ef)
            
            if msa_collection.count() > 0:
                try:
                    # Increase initial n_results to get chunks from all documents
                    msa_results = msa_collection.query(
                        query_texts=[query],
                        n_results=20,  # Increased to get more initial results
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Group results by source document
                    doc_groups = {}
                    if msa_results['documents'][0]:
                        for doc, metadata, distance in zip(
                            msa_results['documents'][0], 
                            msa_results['metadatas'][0],
                            msa_results['distances'][0]
                        ):
                            source = metadata['source']
                            if source not in doc_groups:
                                doc_groups[source] = []
                            doc_groups[source].append((doc, metadata, distance))
                    
                    # Take top 5 most relevant chunks from each document
                    for source, chunks in doc_groups.items():
                        chunks.sort(key=lambda x: x[2])  # Sort by distance (relevance)
                        top_chunks = chunks[:5]  # Get top 5 from each document
                        for doc, metadata, _ in top_chunks:
                            if doc and len(doc.strip()) > 0:
                                results.append({
                                    'text': doc,
                                    'metadata': metadata
                                })
                        st.write(f"Found {len(top_chunks)} relevant chunks from {source}")
                except Exception as e:
                    st.error(f"Error querying MSA documents: {str(e)}")
            else:
                st.warning("No MSA documents found in the collection")

        if doc_type == "NDA" or doc_type == "BOTH":
            nda_collection = client.get_collection("nda_documents", embedding_function=openai_ef)
            
            if nda_collection.count() > 0:
                try:
                    nda_results = nda_collection.query(
                        query_texts=[query],
                        n_results=5,  # Limit to 5 results
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Group results by source document
                    doc_groups = {}
                    if nda_results['documents'][0]:
                        for doc, metadata, distance in zip(
                            nda_results['documents'][0], 
                            nda_results['metadatas'][0],
                            nda_results['distances'][0]
                        ):
                            source = metadata['source']
                            if source not in doc_groups:
                                doc_groups[source] = []
                            doc_groups[source].append((doc, metadata, distance))
                    
                    # Take top 5 most relevant chunks per document
                    for source, chunks in doc_groups.items():
                        chunks.sort(key=lambda x: x[2])  # Sort by distance (relevance)
                        top_chunks = chunks[:5]
                        for doc, metadata, _ in top_chunks:
                            if doc and len(doc.strip()) > 0:
                                results.append({
                                    'text': doc,
                                    'metadata': metadata
                                })
                        st.write(f"Found {len(top_chunks)} relevant chunks from {source}")
                except Exception as e:
                    st.error(f"Error querying NDA documents: {str(e)}")
            else:
                st.warning("No NDA documents found in the collection")

    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        print(f"Search Error details: {str(e)}")
    
    st.write(f"Total documents found: {len(results)}")
    return results

def check_collections():
    try:
        msa_collection = client.get_collection("msa_documents", embedding_function=openai_ef)
        nda_collection = client.get_collection("nda_documents", embedding_function=openai_ef)
        
        st.sidebar.write("Collection Status:")
        st.sidebar.write(f"MSA documents: {msa_collection.count()} chunks")
        st.sidebar.write(f"NDA documents: {nda_collection.count()} chunks")
    except Exception as e:
        st.sidebar.error(f"Error checking collections: {str(e)}")

def debug_collection(collection_name):
    try:
        collection = client.get_collection(collection_name, embedding_function=openai_ef)
        st.sidebar.write(f"\nDebug {collection_name}:")
        st.sidebar.write(f"Collection exists: Yes")
        st.sidebar.write(f"Number of chunks: {collection.count()}")
        
        # Get a sample document to verify content
        sample = collection.query(
            query_texts=["test"],
            n_results=1,
            include=["documents", "metadatas"]
        )
        st.sidebar.write("Sample query successful: Yes")
        
    except Exception as e:
        st.sidebar.error(f"Error debugging {collection_name}: {str(e)}")

def check_prompt_safety(query):
    safety_prompt = """You are a content safety classifier. Evaluate if the following query is safe and related to legal document processing.
    Respond with exactly "SAFE" or "UNSAFE"."""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": safety_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message['content'].strip() == "SAFE"
    except Exception as e:
        st.error(f"Safety check failed: {str(e)}")
        return False

def main():
    st.title("Legal Document Processing System")
    check_collections()
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    
    query = st.text_input("Enter your query:")
    
    if query:
        with st.spinner("Checking query safety..."):
            if not check_prompt_safety(query):
                st.error("⚠️ Query rejected for safety reasons. Please ensure your query is appropriate and related to legal document processing.")
                return
                
        with st.spinner("Analyzing query..."):
            intent = analyze_intent(query)
            st.write(f"Detected intent: Processing {intent} documents")
            
            results = search_documents(query, intent)
            
            if results:
                doc_results = {}
                for result in results:
                    source = result['metadata']['source']
                    if source not in doc_results:
                        doc_results[source] = []
                    doc_results[source].append(result)
                
                # Process each document separately
                for source, doc_chunks in doc_results.items():
                    with st.spinner(f"Processing document: {source}"):
                        try:
                            if source not in st.session_state.processed_data:
                                combined_text = " ".join([chunk['text'] for chunk in doc_chunks])
                                doc_type = doc_chunks[0]['metadata']['type']
                                raw_data = extract_key_terms(combined_text, doc_type)
                                
                                # Store processed data in session state
                                st.session_state.processed_data[source] = {
                                    'raw_data': raw_data,
                                    'doc_type': doc_type,
                                    'flattened_data': flatten_extracted_data(raw_data, source, doc_type)
                                }
                            
                            # Display document data
                            st.subheader(f"Document: {source}")
                            
                            # Create DataFrame for preview
                            df = pd.DataFrame(st.session_state.processed_data[source]['flattened_data'])
                            
                            # Configure columns
                            column_config = {
                                'source_document': st.column_config.TextColumn('Source', width='medium'),
                                'document_type': st.column_config.TextColumn('Type', width='small'),
                                'category': st.column_config.TextColumn('Category', width='medium'),
                                'key_term': st.column_config.TextColumn('Key Term', width='large'),
                                'description': st.column_config.TextColumn('Description', width='large'),
                            }
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📥 Save Directly", key=f"direct_save_{source}"):
                                    save_to_database(
                                        st.session_state.processed_data[source]['flattened_data'],
                                        source,
                                        st.session_state.processed_data[source]['doc_type']
                                    )
                                    st.success(f"✅ Saved original extraction for {source}")
                            
                            with col2:
                                if st.button("✏️ Review & Edit", key=f"review_{source}"):
                                    st.session_state[f"show_editor_{source}"] = True
                            
                            # Show either preview or editor based on state
                            if st.session_state.get(f"show_editor_{source}", False):
                                edited_df = st.data_editor(
                                    df,
                                    column_config=column_config,
                                    num_rows="dynamic",
                                    use_container_width=True,
                                    hide_index=True,
                                    key=f"editor_{source}"
                                )
                                
                                if st.button("💾 Save Edited Version", key=f"save_edited_{source}"):
                                    save_to_database(
                                        edited_df.to_dict('records'),
                                        source,
                                        st.session_state.processed_data[source]['doc_type']
                                    )
                                    st.success(f"✅ Saved edited version for {source}")
                            else:
                                # Show non-editable preview
                                st.dataframe(
                                    df[['source_document', 'document_type', 'category', 'key_term', 'description']],
                                    column_config=column_config,
                                    hide_index=True
                                )
                            
                            st.divider()
                            
                        except Exception as e:
                            st.error(f"Error processing document {source}: {str(e)}")
            else:
                st.warning("No documents found matching your query.")

def save_to_database(data, source, doc_type):
    try:
        # Test connection before operations
        mongo_client.admin.command('ping')
        
        collection = db.msa_data if doc_type == "MSA" else db.nda_data
        try:
            # Ensure data is properly formatted
            if not isinstance(data, list):
                data = [data]
            
            # Add timestamp to each record
            for record in data:
                record['updated_at'] = datetime.datetime.utcnow()
            
            # Delete existing entries for this document
            collection.delete_many({"source_document": source})
            
            # Insert new data one by one with error handling
            successful_inserts = 0
            for record in data:
                try:
                    collection.insert_one(record)
                    successful_inserts += 1
                except pymongo.errors.DuplicateKeyError:
                    continue  # Skip duplicates
            
            st.success(f"✅ Successfully saved {successful_inserts} records for {source}")
            
        except pymongo.errors.PyMongoError as e:
            st.error(f"Database operation error: {str(e)}")
            
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")

def flatten_extracted_data(raw_data, source, doc_type):
    # Flattening logic from the original code
    flattened_data = []
    for category, items in raw_data.items():
        if category in ['effective_date', 'governing_law']:
            # Handle single items
            if category == 'effective_date':
                if items.get('date') and items.get('description'):
                    flattened_data.append({
                        'category': category,
                        'key_term': items['date'],
                        'description': items['description'],
                        'source_document': source,
                        'document_type': doc_type
                    })
            else:
                if items.get('jurisdiction') and items.get('description'):
                    flattened_data.append({
                        'category': category,
                        'key_term': items['jurisdiction'],
                        'description': items['description'],
                        'source_document': source,
                        'document_type': doc_type
                    })
        else:
            # Handle list items
            for item in items:
                if item.get('term') and item.get('description'):
                    flattened_data.append({
                        'category': category,
                        'key_term': item['term'],
                        'description': item['description'],
                        'source_document': source,
                        'document_type': doc_type
                    })
    return flattened_data

if __name__ == "__main__":
    main() 