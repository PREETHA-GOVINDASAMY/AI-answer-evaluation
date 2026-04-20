# =========================================
# 🎓 AI Answer Evaluation System (Offline)
# =========================================

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================================
# UI
# =========================================
st.set_page_config(page_title="AI Evaluator", layout="wide")
st.title("🎓 AI Answer Evaluation System (Word-Based)")

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model_bert = load_model()

# =========================================
# WORD-BASED ANSWER GENERATION
# =========================================
def generate_answer(question, marks):

    topic = question.replace("What is", "").replace("Define", "").strip()

    if marks == 2:
        return f"""{topic} is an important concept in computer science. It helps systems perform tasks efficiently and accurately. It is widely used in modern technology and plays a key role in solving problems."""

    elif marks == 5:
        return f"""{topic} is a fundamental concept in computer science and technology. It is widely used in many applications to improve efficiency and accuracy.

It helps in solving complex problems using structured methods. It reduces human effort and improves productivity.

Applications include automation, data processing, and intelligent systems. It plays an important role in modern computing and real-world problem-solving."""

    elif marks == 10:
        return f"""{topic} is an important concept in computer science that plays a crucial role in modern technology.

It involves principles and methods that help systems process information efficiently. It improves accuracy and reduces manual work.

Advantages include better performance, scalability, and flexibility. It is used in automation, data analysis, and software systems.

Overall, it contributes significantly to modern technological advancements."""

    elif marks == 16:
        return f"""{topic} is a major concept in computer science.

Introduction:
It helps systems solve problems efficiently.

Working:
Input → Processing → Output

Advantages:
• Saves time  
• Improves accuracy  
• Reduces effort  

Applications:
• AI systems  
• Automation  
• Data analysis  

Conclusion:
It plays a key role in modern technology."""

# =========================================
# EVALUATION FUNCTION
# =========================================
def evaluate(model, student, marks):

    emb = model_bert.encode([model, student])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

    # LENGTH SCORING
    model_len = len(model.split())
    student_len = len(student.split())
    ratio = student_len / model_len if model_len > 0 else 0

    if ratio >= 0.9:
        length_score = 1
    elif ratio >= 0.7:
        length_score = 0.8
    elif ratio >= 0.5:
        length_score = 0.6
    elif ratio >= 0.3:
        length_score = 0.4
    else:
        length_score = 0.2

    # KEYWORD ANALYSIS
    model_words = set(model.lower().split()) - ENGLISH_STOP_WORDS
    student_words = set(student.lower().split()) - ENGLISH_STOP_WORDS

    matched = model_words & student_words
    missing = model_words - student_words

    keyword = len(matched) / max(len(model_words), 1)

    # MARKS DISTRIBUTION
    semantic_marks = sim * (marks * 0.5)
    length_marks = length_score * (marks * 0.2)
    keyword_marks = keyword * (marks * 0.3)

    score = round(semantic_marks + length_marks + keyword_marks, 2)

    return score, sim, length_score, keyword, matched, missing

# =========================================
# INPUT
# =========================================
col1, col2 = st.columns(2)

with col1:
    question = st.text_input("❓ Enter Question")
    marks = st.selectbox("Marks", [2, 5, 10, 16])

with col2:
    student_answer = st.text_area("🧑‍🎓 Student Answer")

# =========================================
# EVALUATE
# =========================================
if st.button("Evaluate"):

    if not question or not student_answer:
        st.error("⚠️ Enter all fields")
    else:
        model_answer = generate_answer(question, marks)

        st.subheader("📖 Model Answer")
        st.write(model_answer)

        score, sim, length_score, keyword, matched, missing = evaluate(
            model_answer, student_answer, marks
        )

        st.subheader("📊 Result")
        st.write(f"Score: {score} / {marks}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Semantic", round(sim, 2))
        c2.metric("Length", round(length_score, 2))
        c3.metric("Keyword", round(keyword, 2))

        # Keyword Analysis
        st.subheader("🔍 Keyword Analysis")

        colA, colB = st.columns(2)

        with colA:
            st.write("✅ Matched Keywords")
            st.write(", ".join(list(matched)[:10]) or "None")

        with colB:
            st.write("❌ Missing Keywords")
            st.write(", ".join(list(missing)[:10]) or "None")

        # Feedback
        st.subheader("🧠 Feedback")

        if sim < 0.5:
            st.error("❌ Poor understanding")
        elif sim > 0.8:
            st.success("✅ Excellent understanding")

        if length_score < 0.5:
            st.warning("⚠️ Answer too short")
        else:
            st.success("✅ Good answer length")

        if keyword < 0.5:
            st.warning("⚠️ Missing keywords")

        if sim > 0.75 and keyword > 0.6:
            st.success("🌟 Excellent Answer!")

st.markdown("---")
st.write("🎓 Final Year Project | Offline AI Evaluation System")