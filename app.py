import streamlit as st

# This is the page title — shows in the browser tab
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Main heading on the page
st.title("🤖 RAG Chatbot")
st.write("Ask me anything about your documents.")

# Text input box — where the user types their question
user_input = st.text_input("Your question:", placeholder="e.g. What is the refund policy?")

# Button — clicking this will trigger the logic
if st.button("Ask"):
    if user_input.strip() == "":
        # If user clicks Ask without typing anything
        st.warning("Please enter a question first.")
    else:
        # For now just echo back — Claude + ChromaDB gets wired in later
        st.success(f"You asked: {user_input}")
        st.info("Answer will appear here once we wire up Claude and ChromaDB.")