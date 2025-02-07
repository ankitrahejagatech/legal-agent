import pymongo
import streamlit as st

def setup_mongodb():
    client = pymongo.MongoClient(st.secrets["MONGODB_URI"])
    db = client.legal_documents
    
    # Create collections if they don't exist
    try:
        db.create_collection("msa_data")
    except pymongo.errors.CollectionInvalid:
        pass
    
    try:
        db.create_collection("nda_data")
    except pymongo.errors.CollectionInvalid:
        pass
    
    # Clean up any null values in existing data
    for collection in [db.msa_data, db.nda_data]:
        collection.update_many(
            {"$or": [
                {"category": None},
                {"key_term": None}
            ]},
            {"$set": {
                "category": "uncategorized",
                "key_term": "undefined"
            }}
        )
        
        # Drop existing indexes
        collection.drop_indexes()
        
        # Create new compound unique index
        collection.create_index(
            [
                ("source_document", 1),
                ("category", 1),
                ("key_term", 1)
            ],
            unique=True,
            sparse=True  # Ignore documents that don't have all fields
        )
    
    return client

if __name__ == "__main__":
    try:
        client = setup_mongodb()
        print("MongoDB collections and indexes created successfully!")
    except Exception as e:
        print(f"Error setting up MongoDB: {str(e)}") 