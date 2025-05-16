import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Incident Insights Dashboard", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center;'>
        🚨 Incident Insights Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

# Load data directly from files
trend_df = pd.read_excel("sample_incident_trends.xlsx")
sim_df = pd.read_excel("sample_resolution_similarity.xlsx")

# Convert date column to datetime
trend_df['date'] = pd.to_datetime(trend_df['date'])

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
notes = sim_df['matched_note'].tolist()
note_embeddings = model.encode(notes, convert_to_tensor=True)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Incident Trends Over Time")
    fig = px.line(trend_df, x='date', y='incident_count', color='assignment_group',
                  title="Monthly Incident Volume per Assignment Group",
                  labels={'incident_count': 'Incident Count'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔍 Resolution Note Similarity Search")
    query_input = st.text_area("Enter a new resolution note:", "Application crash resolved by upgrading Java runtime.")

    if query_input:
        query_embedding = model.encode([query_input], convert_to_tensor=True)
        similarity_scores = cosine_similarity(query_embedding, note_embeddings)[0]
        sim_df['similarity'] = similarity_scores
        top_similar = sim_df.sort_values(by='similarity', ascending=False).head(5)[['matched_incident', 'matched_note', 'similarity']]
        top_similar.columns = ['Incident ID', 'Matched Note', 'Similarity Score']

        st.markdown("### 🔗 Top Matching Past Incidents")
        st.dataframe(top_similar, use_container_width=True)
