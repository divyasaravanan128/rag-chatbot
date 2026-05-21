# reingest.py
import chromadb

# 1. Clear the collection
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection("documents")
print("✓ Collection cleared")

# 2. Remind you to delete the file from /docs
print("✓ Now delete your file from /docs, then re-upload via the Streamlit sidebar")