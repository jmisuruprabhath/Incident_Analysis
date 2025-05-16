import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from utils.scraper import load_website_data
from utils.vector_store import setup_vectorstore

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1) Build local HF pipeline
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
hf_pipe   = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device=-1,            # CPU; set to 0 for GPU
)

# 2) Wrap in LangChain LLM
hf_llm = HuggingFacePipeline(pipeline=hf_pipe)

# 3) Embeddings & VectorStore factory
embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={"device": "cpu"}
                )

# Streamlit UI Setup
st.set_page_config(page_title="Web QA Chatbot", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
        color: white;
    }
    .response-card {
        background-color: #444;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 16px;
        display: flex;
        flex-direction: column;
    }
    .user-input {
        background-color: #333;
        color: white;
        padding: 10px;
        border-radius: 10px;
        font-size: 16px;
        display: flex;
        justify-content: flex-start;
    }
    .chat-history {
        background-color: #222;
        padding: 15px;
        height: 80vh;
        overflow-y: auto;
        width: 350px;
        position: absolute;
        right: 0;
        top: 20px;
        border-radius: 10px;
        font-size: 14px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .chat-container {
        width: calc(100% - 370px);
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .input-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .latest-response {
        margin-top: 20px;
        background-color: #333;
        color: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid green;
    }
    .text-input {
        font-size: 14px;
    }
    .gap {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Web-based QA Chatbot")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa" not in st.session_state:
    st.session_state.qa = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# hf_api_key = "key"#os.getenv("HUGGINGFACE_API_TOKEN")

# Layout for left side (prompting, answers, input) and right side (chat history)
col1, col2 = st.columns([3, 1])  # 3 parts for left section, 1 part for history

# Chat UI in the left column
with col1:
    # Input for the website URL
    url = st.text_input("Enter the website URL:", "", key="url", help="Enter the website URL to load the data.", max_chars=200)
    
    # Button to load website data
    if st.button("Load Website Data") and url:
        with st.spinner("🔄 Fetching and processing data..."):
            try:
                loader = WebBaseLoader(url)
                documents = loader.load()

                # Load website content
                documents = load_website_data(url)
                vectorstore = setup_vectorstore(documents)

                # llm = HuggingFaceEndpoint(
                #     repo_id="google/flan-t5-large",
                #     huggingfacehub_api_token=hf_api_key,
                #     # any other kwargs you need:
                #     temperature=0,
                #     max_new_tokens=512,
                # )

                # # Retrieval-based QA System
                # qa = RetrievalQA.from_chain_type(
                #     llm=llm, #HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token=hf_api_key), 
                #     chain_type="stuff",
                #     retriever=vectorstore.as_retriever()
                # )
                qa = RetrievalQA.from_chain_type(
                    llm=hf_llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )

                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.qa = qa

                st.success("✅ Data successfully loaded. You can now ask questions!")

            except Exception as e:
                st.error(f"❌ Error loading website: {e}")

    # Input for query
    if st.session_state.qa:
        query = st.text_input("💬 Ask a question:", "", key="query", help="Ask a question about the website.", max_chars=150)

        # Button to get the answer
        if st.button("Get Answer") and query:
            print('Start Query Test')
            with st.spinner("🔎 Searching for the best answer..."):
                try:
                    print('Start Query Before')
                    #response = st.session_state.qa.run(query) #qa.run(query) 
                    response = qa.run(query) 
                    print('Start Query After')
                    
                    # Append chat history
                    st.session_state.chat_history.append(("You", query))
                    st.session_state.chat_history.append(("Bot", response))

                    # Display the latest answer
                    st.session_state.latest_response = response

                except Exception as e:
                    st.error(f"❌❌ Error generating response: {e}")

    # Display the latest response (if available)
    if "latest_response" in st.session_state:
        st.markdown(f"<div class='latest-response'><b>🤖 </b>{st.session_state.latest_response}</div>", unsafe_allow_html=True)

    # Space between chat history and prompts
    st.markdown("<div class='gap'></div>", unsafe_allow_html=True)

# Chat history in the right column
with col2:
    st.markdown("<h3 style='text-align: center;'>Chat History</h3>", unsafe_allow_html=True)
    with st.container():
        # Display all chat history in the right panel, grouping user input and bot response together
        for i in range(0, len(st.session_state.chat_history), 2):
            user_message = st.session_state.chat_history[i]
            bot_message = st.session_state.chat_history[i + 1] if i + 1 < len(st.session_state.chat_history) else None
            
            if user_message and bot_message:
                st.markdown(f"""
                    <div class="response-card">
                        <b>🧑‍💻 You: </b>{user_message[1]}
                        <b>🤖 Bot: </b>{bot_message[1]}
                    </div>
                """, unsafe_allow_html=True)
