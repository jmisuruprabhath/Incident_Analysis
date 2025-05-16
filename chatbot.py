import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

import time
from datetime import datetime
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Model & Embeddings Setup
# -------------------------------
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
hf_pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device=-1  # CPU; use 0 if GPU is available
)
hf_llm = HuggingFacePipeline(pipeline=hf_pipe)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

# -------------------------------
# CSV Loader Function
# -------------------------------
def load_csv_data(file_path: str) -> list[Document]:
    df = pd.read_csv(file_path)
    docs = []
    for _, row in df.iterrows():
        content = f"""Incident Number: {row['incident_number']}
                        Assignment Group: {row['assignment_group']}
                        Status: {row['status']}
                        Priority: {row['Priority']}
                        Created Date: {row['created_date']}
                        MTTR: {row['MTTR']}
                        Resolution Notes: {row['resolution_notes']}
                        Description: {row['incident_description']}"""
        docs.append(Document(page_content=content))
    return docs

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Incident Insights Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .main > div:first-child {
            padding-top: 0rem;
        }
        h1 {
            margin-top: 0px;
            padding-top: 0px;
        }
    </style>
    <h1 style='text-align: center; margin-top: 0px; padding-top: 0px;'>
        🚨 Incident Insights Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

# Load data directly from files
trend_df = pd.read_csv("incident_forecast_output.csv")
sim_df = pd.read_excel("sample_resolution_similarity.xlsx")
inc_df = pd.read_csv("incident_dataset.csv")

# Convert date column to datetime
#trend_df['date'] = pd.to_datetime(trend_df['date'])

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
notes = sim_df['matched_note'].tolist()
note_embeddings = model.encode(notes, convert_to_tensor=True)
# st.title("📁 CSV-based QA Chatbot")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa" not in st.session_state:
    st.session_state.qa = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hardcoded file path
file_path = "incident_dataset.csv"  # Replace with your actual path

if st.session_state.vectorstore is None and st.session_state.qa is None:
    with st.spinner("🔄 Embedding data and preparing data..."):
        try:
            documents = load_csv_data(file_path)
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(docs, embedding=embeddings)

            qa = RetrievalQA.from_chain_type(
                llm=hf_llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.qa = qa

            st.success("✅ DB embedding loaded. Ask your questions!")

        except Exception as e:
            st.error(f"❌ Error loading CSV data: {e}")

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1]) 

# with col2:
#     if st.session_state.qa:
#         query = st.text_input("💬 Ask a question about the incident data:", "", key="query")
#         if st.button("Get Answer") and query:
#             with st.spinner("🔎 Searching for the best answer..."):
#                 try:
#                     response = st.session_state.qa.run(query)
#                     st.session_state.chat_history.append(("You", query))
#                     st.session_state.chat_history.append(("Bot", response))
#                     st.session_state.latest_response = response
#                 except Exception as e:
#                     st.error(f"❌ Error generating response: {e}")

#     if "latest_response" in st.session_state:
#         st.markdown(f"**🤖 Answer:** {st.session_state.latest_response}")

# with col2:
#     st.markdown("### 🕘 Chat History")
#     for i in range(0, len(st.session_state.chat_history), 2):
#         user = st.session_state.chat_history[i]
#         bot = st.session_state.chat_history[i + 1] if i + 1 < len(st.session_state.chat_history) else None
#         if bot:
#             st.markdown(f"- **You:** {user[1]}\n- **Bot:** {bot[1]}")

with col1:
    st.subheader("💬 LogBuster Agent")
    if st.session_state.qa:
        query = st.text_input("💬 Ask a question about the incident data:", "", key="query")
        if st.button("Get Answer") and query:
            with st.spinner("🔎 Searching for the best answer..."):
                try:
                    response = st.session_state.qa.run(query)
                    st.session_state.chat_history.append(("You", query))
                    st.session_state.chat_history.append(("Bot", response))
                    st.session_state.chat_history = st.session_state.chat_history[-8:]

                    st.session_state.latest_response = response
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")

    if "latest_response" in st.session_state:
        st.markdown(f"**🤖 Answer:** {st.session_state.latest_response}")

    # st.markdown("""<h3 style='text-align: center;'>🕘 Chat History</h3>""", unsafe_allow_html=True)
    for i in range(0, len(st.session_state.chat_history), 2):
        user = st.session_state.chat_history[i]
        bot = st.session_state.chat_history[i + 1] if i + 1 < len(st.session_state.chat_history) else None
        if user and bot:
            st.markdown(f"""
                <div style='background-color: #808080; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <div style='margin-bottom: 5px;'><b>🧑‍💻 You:</b> {user[1]}</div>
                </div>

                 <div style='background-color: #3d0099; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                   <div> <b>🤖 Bot: </b>{bot[1]} </div>
                </div>
            """, unsafe_allow_html=True)


with col2:
    inc_df['created_date'] = pd.to_datetime(inc_df['created_date'], errors='coerce')
    inc_df['year'] = inc_df['created_date'].dt.year
    inc_df['month'] = inc_df['created_date'].dt.month_name()

    if 'resolution_notes' in inc_df.columns:
        inc_df['category'] = inc_df['resolution_notes'].str.split(':').str[0].str.strip()

    years = sorted(inc_df['year'].dropna().unique(), reverse=True)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    months = sorted(inc_df['month'].dropna().unique(), key=lambda x: month_order.index(x))
    assignment_groups = inc_df['assignment_group'].dropna().unique()
    priorities = inc_df['Priority'].dropna().unique()
    statuses = inc_df['status'].dropna().unique()
    categories = sorted(inc_df['category'].dropna().unique())

    # Filters
    row1 = st.columns(6)
    with row1[0]:
        selected_year = st.selectbox("Select Year", years)
    with row1[1]:
        selected_month = st.selectbox("Select Month", months)
    with row1[2]:
        selected_group = st.multiselect("Assignment Group", assignment_groups, default=list(assignment_groups))
    with row1[3]:
        selected_priority = st.multiselect("Priority", priorities, default=list(priorities))
    with row1[4]:
        selected_status = st.multiselect("Status", statuses, default=list(statuses))
    with row1[5]:
        selected_category = st.multiselect("Category", categories, default=categories)

    # Filter Data
    filtered_df = inc_df[
        (inc_df['year'] == selected_year) &
        (inc_df['month'] == selected_month) &
        (inc_df['assignment_group'].isin(selected_group)) &
        (inc_df['Priority'].isin(selected_priority)) &
        (inc_df['status'].isin(selected_status)) &
        (inc_df['category'].isin(selected_category))
    ]

    # KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Incidents", len(filtered_df))
    with kpi2:
        st.metric("Open Incidents", len(filtered_df[filtered_df['status'].str.lower() == 'open']))
    with kpi3:
        st.metric("Completed Incidents", len(filtered_df[filtered_df['status'].str.lower() == 'completed']))
    with kpi4:
        st.metric("In Progress", len(filtered_df[filtered_df['status'].str.lower() == 'in-progress']))
    with kpi5:
        st.metric("Cancelled", len(filtered_df[filtered_df['status'].str.lower() == 'cancelled']))

    st.subheader("📊 Incident Distribution by Priority, Assignment Group & Category")
    # Row of 3 Charts
    chart1, chart2, chart3 = st.columns(3)
    
    with chart1:
        if not filtered_df.empty:
            priority_counts = filtered_df['Priority'].value_counts().reset_index()
            priority_counts.columns = ['Priority', 'Count']
            fig_pie_priority = px.pie(
                priority_counts,
                names='Priority',
                values='Count',
                title='Incident Count by Priority',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie_priority.update_traces(
                textinfo='label+value',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            )
            st.plotly_chart(fig_pie_priority, use_container_width=True)
        else:
            st.warning("No data available for pie chart.")

    with chart2:
        if not filtered_df.empty:
            group_counts = filtered_df['assignment_group'].value_counts().reset_index()
            group_counts.columns = ['Assignment Group', 'Incident Count']
            fig_group_bar = px.bar(
                group_counts,
                x='Assignment Group',
                y='Incident Count',
                title='Incident Count by Assignment Group',
                text='Incident Count',
                color='Assignment Group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_group_bar.update_traces(textposition='outside')
            fig_group_bar.update_layout(
                xaxis_title='Assignment Group',
                yaxis_title='Incident Count',
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig_group_bar, use_container_width=True)
        else:
            st.warning("No data available for bar chart.")

    with chart3:
        if not filtered_df.empty:
            category_counts = filtered_df['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            fig_pie_category = px.pie(
                category_counts,
                names='Category',
                values='Count',
                title='Incident Count by Category',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_pie_category.update_traces(
                textinfo='label+value',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            )
            st.plotly_chart(fig_pie_category, use_container_width=True)
        else:
            st.warning("No data available for category pie chart.")

    # # Line Chart for Trends
    # st.subheader("📈 Incident Trends Over Time")
    # fig = px.line(
    #     trend_df.sort_values(by='month'),
    #     x='month',
    #     y='incident_count',
    #     color='assignment_group',
    #     title="Yearly Trend per Assignment Group",
    #     labels={'incident_count': 'Incident Count', 'assignment_group': 'Assignment Group'}
    # )
    # st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("🔍 Resolution Note Search")
    query_input = st.text_area("Enter a new resolution note:", "Application crash resolved by upgrading Java runtime.")

    if query_input:
        query_embedding = model.encode([query_input], convert_to_tensor=True)
        similarity_scores = cosine_similarity(query_embedding, note_embeddings)[0]
        sim_df['similarity'] = similarity_scores
        top_similar = sim_df.sort_values(by='similarity', ascending=False).head(5)[['matched_incident', 'matched_note', 'similarity']]
        top_similar.columns = ['Incident ID', 'Matched Note', 'Similarity Score']

        st.markdown("### 🔗 Top Matching Past Incidents")
        st.dataframe(top_similar, use_container_width=True)

    #Line Chart for Trends
    # 📈 Line Chart for Trends with future date highlighted
st.subheader("📈 Incident Trends Over Time")

# Convert 'month' to datetime if it's not already
trend_df['month'] = pd.to_datetime(trend_df['month'], errors='coerce')

# Get today's date
today = pd.to_datetime(datetime.today().date())

# Split the dataframe into past and future
past_df = trend_df[trend_df['month'] <= today]
future_df = trend_df[trend_df['month'] > today]

# Create the figure
fig = px.line(
    past_df.sort_values(by='month'),
    x='month',
    y='incident_count',
    color='assignment_group',
    title="Monthly Incident Volume per Assignment Group",
    labels={'incident_count': 'Incident Count', 'assignment_group': 'Assignment Group'}
)

# Add future data with a dashed line
if not future_df.empty:
    for group in future_df['assignment_group'].unique():
        future_group_df = future_df[future_df['assignment_group'] == group].sort_values(by='month')
        fig.add_scatter(
            x=future_group_df['month'],
            y=future_group_df['incident_count'],
            mode='lines+markers',
            name=f"{group} (Forecast)",
            line=dict(dash='dash'),
            marker=dict(symbol='circle'),
            showlegend=True
        )

# Display the chart
st.plotly_chart(fig, use_container_width=True)