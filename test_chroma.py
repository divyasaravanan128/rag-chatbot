import chromadb

# Create a local ChromaDB client
# This stores data in memory — nothing is saved to disk yet
client = chromadb.Client()

# Create a collection — think of this like a table in a normal database
# This is where your document chunks will live
collection = client.create_collection(name="my_first_collection")

# Add 3 documents to the collection
# ids are mandatory — ChromaDB needs a unique identifier for each chunk
collection.add(
    documents=[
        "The refund policy allows customers to return items within 30 days.",
        "Our office is open Monday to Friday, 9am to 6pm.",
        "To reset your password, click Forgot Password on the login page."
    ],
    ids=["doc1", "doc2", "doc3"]
)

print("✅ 3 documents added to ChromaDB")

# Now query the collection with a question
# ChromaDB will find the most semantically similar document
results = collection.query(
    query_texts=["How is the weather today?"],
    n_results=1  # return only the top match
)

print("\n🔍 Query: How is the weather today?")
print("📄 Best match:", results["documents"][0][0])