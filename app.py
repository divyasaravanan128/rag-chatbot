import streamlit as st
from dotenv import load_dotenv
import chromadb

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Load API key
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot")
st.write("Ask me anything about your documents.")

# --- Session state (fix duplicate UI issue) ---
if "response" not in st.session_state:
    st.session_state.response = None

# Input
user_input = st.text_input("Your question:", placeholder="e.g. What is the refund policy?")

# --- Initialize ChromaDB ---
client = chromadb.Client()
collection = client.get_or_create_collection(name="my_docs")

# Add docs only once
if collection.count() == 0:
    collection.add(
        documents=[
            "The refund policy allows customers to return items within 30 days.",
            "Our office is open Monday to Friday, 9am to 6pm.",
            "To reset your password, click Forgot Password on the login page."
        ],
        ids=["doc1", "doc2", "doc3"]
    )

# --- Initialize Claude ---
llm = ChatAnthropic(model="claude-sonnet-4-5")

# --- Button logic ---
if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please enter a question first.")
    else:
        results = collection.query(
            query_texts=[user_input],
            n_results=2
        )

        docs = results["documents"][0]
        distances = results["distances"][0]

        best_distance = distances[0]
        THRESHOLD = 1.2

        if best_distance > THRESHOLD:
            st.session_state.response = "I don’t have enough relevant information to answer this."
        else:
            context = "\n".join(docs)

            prompt = f"""
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {user_input}
            """

            response = llm.invoke([HumanMessage(content=prompt)])
            st.session_state.response = response.content

# --- Display response (outside button) ---
if st.session_state.response:
    st.subheader("Answer")
    st.write(st.session_state.response)