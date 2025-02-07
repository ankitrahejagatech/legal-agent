import os
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
import streamlit as st

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 0:
                    text += page_text + "\n"
            except Exception as e:
                print(f"Error extracting text from page in {file_path}: {str(e)}")
                continue
                
        # Basic validation of extracted text
        if not text or len(text.strip()) < 50:  # Minimum 50 characters
            print(f"Warning: Insufficient text extracted from {file_path}")
            return None
            
        # Remove any non-ASCII characters and normalize whitespace
        text = ' '.join(''.join(char if ord(char) < 128 else ' ' for char in text).split())
        return text
    except Exception as e:
        print(f"Failed to process PDF {file_path}: {str(e)}")
        return None

def create_embeddings():
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use OpenAI embeddings with text-embedding-ada-002 model (1536 dimensions)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-ada-002"
    )
    
    # Create collections for MSAs and NDAs
    msa_collection = client.get_or_create_collection(
        name="msa_documents",
        embedding_function=openai_ef
    )
    
    nda_collection = client.get_or_create_collection(
        name="nda_documents",
        embedding_function=openai_ef
    )
    
    # Process MSA documents with validation
    msa_dir = "MSAs"
    for filename in os.listdir(msa_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(msa_dir, filename)
            document_text = extract_text_from_pdf(file_path)
            
            if document_text is None:
                print(f"Skipping problematic MSA file: {filename}")
                continue
                
            try:
                # Split text into smaller chunks
                chunks = [document_text[i:i+1000] for i in range(0, len(document_text), 1000)]
                
                # Validate chunks
                valid_chunks = []
                valid_ids = []
                valid_metadatas = []
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) >= 50:  
                        valid_chunks.append(chunk)
                        valid_ids.append(f"{filename}_{i}")
                        valid_metadatas.append({"source": filename, "chunk": i, "type": "MSA"})
                
                if valid_chunks:
                    msa_collection.add(
                        documents=valid_chunks,
                        ids=valid_ids,
                        metadatas=valid_metadatas
                    )
                    print(f"Successfully processed MSA document: {filename}")
                else:
                    print(f"No valid chunks found in MSA file: {filename}")
                    
            except Exception as e:
                print(f"Error adding MSA document {filename} to collection: {str(e)}")
                continue
    
    # Process NDA documents
    nda_dir = "NDAs"
    for filename in os.listdir(nda_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(nda_dir, filename)
            document_text = extract_text_from_pdf(file_path)
            
            # Split text into chunks of ~1000 characters
            chunks = [document_text[i:i+1000] for i in range(0, len(document_text), 1000)]
            
            # Create IDs and metadata for each chunk
            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk": i, "type": "NDA"} for i in range(len(chunks))]
            
            # Add to collection
            nda_collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Processed NDA document: {filename}")

if __name__ == "__main__":
    create_embeddings()
    print("Embeddings created successfully!") 