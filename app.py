import streamlit as st
import joblib

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("models/nb_spam_model.pkl")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📌 About Project")
st.sidebar.markdown(
    """
**Spam Email Detection System**

- Built using **NLP & Machine Learning**
- Model: **TF-IDF + Naive Bayes**
- Dataset: SMS Spam Collection
- Output: Spam / Not Spam with confidence

👨‍💻 *Portfolio Project*
"""
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by You 🚀")

# ---------------------------
# Main Title
# ---------------------------
st.markdown(
    "<h1 style='text-align: center;'>📧 Spam Email Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Paste an email or message below to check whether it is spam."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------
# Input Box
# ---------------------------
user_input = st.text_area(
    "✉️ Email / Message Text",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize. Click now..."
)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔍 Check Spam", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]

        st.markdown("---")

        # ---------------------------
        # Result Display
        # ---------------------------
        if prediction == 1:
            spam_prob = probabilities[1] * 100
            st.error("🚨 **This message is SPAM**")
            st.write(f"**Spam Confidence:** {spam_prob:.2f}%")
            st.progress(int(spam_prob))
        else:
            ham_prob = probabilities[0] * 100
            st.success("✅ **This message is NOT SPAM**")
            st.write(f"**Not Spam Confidence:** {ham_prob:.2f}%")
            st.progress(int(ham_prob))

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Built with  using Python, NLP & Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
