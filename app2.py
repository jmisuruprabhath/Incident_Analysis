import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------- Configuration -----------------
st.set_page_config(page_title="Web QA Chatbot", layout="wide")

# ----------------- Session State -----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa" not in st.session_state:
    st.session_state.qa = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- Build Local HF Pipeline -----------------
@st.experimental_singleton
def load_local_pipeline(model_id: str = "google/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        device=0,  # use -1 for CPU only
    )

hf_pipe = load_local_pipeline()
hf_llm = HuggingFacePipeline(pipeline=hf_pipe)

# ----------------- UI Layout -----------------
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🔍 Web-based QA Chatbot")
    url = st.text_input(
        "Enter the website URL:", "", key="url",
        help="Enter the website URL to load the data.", max_chars=200
    )

    if st.button("Load Website Data") and url:
        with st.spinner("🔄 Fetching and processing data..."):
            try:
                # Load and parse documents
                loader = WebBaseLoader(url)
                documents = loader.load()

                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name="web_qa_collection"
                )

                # Build RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=hf_llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )

                # Store in session
                st.session_state.vectorstore = vectorstore
                st.session_state.qa = qa_chain
                st.success("✅ Data successfully loaded. You can now ask questions!")
            except Exception as e:
                st.error(f"❌ Error loading website: {e}")

    if st.session_state.qa:
        query = st.text_input(
            "💬 Ask a question:", "", key="query",
            help="Ask a question about the website.", max_chars=150
        )
        if st.button("Get Answer") and query:
            with st.spinner("🔎 Searching for the best answer..."):
                try:
                    response = st.session_state.qa.run(query)
                    # Append to history
                    st.session_state.chat_history.append(("You", query))
                    st.session_state.chat_history.append(("Bot", response))
                    st.session_state.latest_response = response
                except Exception as e:
                    st.error(f"❌❌ Error generating response: {e}")

        if "latest_response" in st.session_state:
            st.markdown(
                f"<div style='background-color:#333;color:white;padding:15px;border-radius:10px;'>"
                f"<b>🤖</b> {st.session_state.latest_response}</div>",
                unsafe_allow_html=True
            )

with col2:
    st.markdown("# Chat History")
    for i in range(0, len(st.session_state.chat_history), 2):
        user_msg = st.session_state.chat_history[i][1]
        bot_msg = st.session_state.chat_history[i+1][1] if i+1 < len(st.session_state.chat_history) else ""
        st.markdown(
            f"<div style='background-color:#444;color:white;padding:10px;border-radius:8px;margin-bottom:8px;'>"
            f"<b>🧑‍💻 You:</b> {user_msg}<br/><b>🤖 Bot:</b> {bot_msg}</div>",
            unsafe_allow_html=True
        )
