import streamlit as st
import numpy as np
import pickle
import re
import os

st.set_page_config(page_title="NER Demo", layout="wide")

@st.cache_data
def load_data():
    try:
        with open("token2idx.pkl", "rb") as f:
            token2idx = pickle.load(f)
        with open("idx2tag.pkl", "rb") as f:
            idx2tag = pickle.load(f)
        return token2idx, idx2tag
    except:
        return {}, {}

st.title("🔍 NER Model Demo")
st.markdown("**Works even without TensorFlow!**")

# Check files
files = ["token2idx.pkl", "idx2tag.pkl"]
missing = [f for f in files if not os.path.exists(f)]
if missing:
    st.error(f"❌ Upload: {', '.join(missing)}")
    st.stop()

token2idx, idx2tag = load_data()
st.success("✅ Ready!")

# Input
col1, col2 = st.columns([3,1])
with col1:
    text = st.text_area("Enter text:", "Hi Divyasri from India wants Google job.", height=100)
with col2:
    max_len = st.slider("Max length", 50, 150, 100)

if st.button("🔮 PREDICT", type="primary"):
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Demo predictions (your real model logic here later)
    tags = ['O', 'B-PER', 'O', 'B-PER', 'O', 'B-LOC', 'O', 'B-ORG', 'O']
    entities = list(zip(words[:10], tags[:10]))
    
    # Table
    st.subheader("📋 Results")
    st.table(entities)
    
    # Highlight
    st.subheader("🎨 Highlighted")
    for word, tag in entities:
        if tag != 'O':
            st.markdown(f"**{word}** ({tag})")
        else:
            st.write(word)
