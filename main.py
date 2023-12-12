import streamlit as st
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Streamlit App
st.title("Aplikasi Ringkasan Berita Online")
st.write("Muhammad Adam Zaky Jiddyansah")

# Text Input Area
berita_input = st.text_area("Masukkan teks berita di sini:")

# Preprocessing Function
def preprocessing(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s.]', '', text)
    text = text.lower()

    stop_words = set(stopwords.words('indonesian'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]

    preprocessing_text = ' '.join(filtered_words)

    return preprocessing_text

# Preprocess Input
berita_input_preprocessed = preprocessing(berita_input)

# Tokenize Sentences
kalimat_input = nltk.sent_tokenize(berita_input_preprocessed)

# Check if kalimat_input is not empty
if not kalimat_input:
    st.warning("Masukkan teks berita terlebih dahulu.")
else:
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2)
    tfidf_input = tfidf_vectorizer.fit_transform(kalimat_input)

    # Cosine Similarity
    cosine_input = cosine_similarity(tfidf_input, tfidf_input)

    # Graph Creation
    G_input = nx.DiGraph()

    # Add nodes to the graph
    for i in range(len(cosine_input)):
        G_input.add_node(i)

    # Add edges based on similarity
    for i in range(len(cosine_input)):
        for j in range(len(cosine_input)):
            similarity = cosine_input[i][j]
            if similarity > 0.1 and i != j:
                G_input.add_edge(i, j)

    # Eigenvector Centrality
    eigenvector_input = nx.eigenvector_centrality(G_input)
    sorted_eigenvector_input = sorted(eigenvector_input.items(), key=lambda x: x[1], reverse=True)

    # Top 3 Nodes
    top3_nodes_input = [node for node, _ in sorted_eigenvector_input[:3]]
    ringkasan_input = " ".join([kalimat_input[node] for node in top3_nodes_input])

    # Display Results
    st.subheader("Ringkasan Berita:")
    st.write(ringkasan_input)

    # Visualization (Graph)
    # st.subheader("Graf Kesamaan Kalimat:")
    # pos_input = nx.spring_layout(G_input)
    # nx.draw_networkx_nodes(G_input, pos_input, node_size=100, node_color='salmon')
    # nx.draw_networkx_edges(G_input, pos_input, edge_color='red', arrows=True)
    # nx.draw_networkx_labels(G_input, pos_input)
    # st.pyplot(plt)

    # # Eigenvector Centrality
    # st.subheader("Nilai Eigenvector Centrality:")
    # for node, eigenvector_input in sorted_eigenvector_input:
    #     st.write(f"Node {node}: {eigenvector_input:.4f}")
