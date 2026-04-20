# =========================================
# 🎓 OFFLINE AI EVALUATION SYSTEM
# =========================================

import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

# =========================================
# LOAD MODELS
# =========================================
@st.cache_resource
def load_models():
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    return bert, nlp

model_bert, nlp = load_models()

# =========================================
# LOAD DATASET
# =========================================
df = pd.read_csv("dataset.csv")

# =========================================
# FUNCTION: GET MODEL ANSWER
# =========================================
def get_model_answer(question):
    match = df[df['question'] == question]
    if not match.empty:
        return match.iloc[0]['model_answer']
    return "No model answer found"

# =========================================
# FUNCTION: SEMANTIC SIMILARITY
# =========================================
def semantic_score(model_ans, student_ans):
    emb = model_bert.encode([model_ans, student_ans])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

# =========================================
# FUNCTION: TF-IDF KEYWORD SCORE
# =========================================
def keyword_score(model_ans, student_ans):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([model_ans, student_ans])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# =========================================
# FUNCTION: CONCEPT EXTRACTION
# =========================================
def concept_score(model_ans, student_ans):
    doc1 = nlp(model_ans)
    doc2 = nlp(student_ans)

    concepts1 = set([ent.text.lower() for ent in doc1.ents])
    concepts2 = set([ent.text.lower() for ent in doc2.ents])

    if len(concepts1) == 0:
        return 0

    return len(concepts1 & concepts2) / len(concepts1)

# =========================================
# FINAL EVALUATION
# =========================================
def evaluate(model_ans, student_ans, marks):

    sim = semantic_score(model_ans, student_ans)
    key = keyword_score(model_ans, student_ans)
    concept = concept_score(model_ans, student_ans)

    final = (0.5 * sim) + (0.3 * key) + (0.2 * concept)
    score = round(final * marks, 2)

    return score, sim, key, concept

# =========================================
# UI
# =========================================
st.title("🎓 Offline AI Answer Evaluation")

question = st.selectbox("Select Question", df['question'])
student_answer = st.text_area("Enter Student Answer")
marks = int(df[df['question'] == question]['marks'].values[0])

# =========================================
# EVALUATE
# =========================================
if st.button("Evaluate"):

    model_ans = get_model_answer(question)

    st.subheader("📖 Model Answer")
    st.write(model_ans)

    score, sim, key, concept = evaluate(model_ans, student_answer, marks)

    st.subheader("📊 Result")
    st.write(f"Score: {score} / {marks}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Semantic", round(sim, 2))
    col2.metric("Keyword", round(key, 2))
    col3.metric("Concept", round(concept, 2))

    # Feedback
    st.subheader("🧠 Feedback")

    if sim < 0.5:
        st.write("❌ Poor understanding")

    if concept < 0.5:
        st.write("⚠️ Missing important concepts")

    if key < 0.5:
        st.write("⚠️ Weak keyword coverage")

    if sim > 0.8:
        st.write("✅ Excellent answer")