# 📧 Spam Email Detection System

An end-to-end **Spam Email Detection System** built using **Natural Language Processing (NLP)** and **Machine Learning**, with a clean **Streamlit-based web interface** for real-time prediction.

---

## 🚀 Project Overview

Spam messages are a common problem in digital communication.  
This project classifies messages as **Spam** or **Not Spam** by analyzing textual patterns using NLP techniques and a machine learning model.

The project demonstrates the **complete ML workflow**:
- Data preprocessing
- Feature extraction
- Model training & evaluation
- Model saving
- Interactive web application (local)

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries & Tools:**
  - Pandas, NumPy
  - NLTK (text preprocessing)
  - Scikit-learn
  - Streamlit (web interface)
  - Joblib (model persistence)

---

## 📂 Project Structure  



spam-email-detector/
│
├── data/
│ └── spam.csv # Dataset
│
├── models/
│ └── nb_spam_model.pkl # Trained ML model
│
├── app.py # Streamlit web app
├── train.py # Model training script
├── requirements.txt # Project dependencies
├── README.md # Project documentation
│
├── .gitignore # Files to ignore in Git
└── venv/ # Virtual environment (NOT pushed to GitHub)


---

## 📊 Dataset

- **Dataset:** SMS Spam Collection Dataset  
- **Description:**  
  A real-world dataset containing SMS messages labeled as:
  - `ham` → Not Spam  
  - `spam` → Spam  

---

## 🔍 Workflow

1. **Data Loading**
   - Loaded dataset and removed unnecessary columns.

2. **Text Preprocessing**
   - Converted text to lowercase  
   - Removed URLs, punctuation, and numbers  
   - Removed stopwords using NLTK  

3. **Feature Extraction**
   - Applied **TF-IDF Vectorization** to convert text into numerical features.

4. **Model Training**
   - Trained a **Naive Bayes classifier**, which performs well for text classification.

5. **Model Evaluation**
   - Evaluated using accuracy, precision, recall, and F1-score.

6. **Model Saving**
   - Saved the trained model using `joblib` for reuse.

7. **Web Application**
   - Built a Streamlit web app for real-time spam detection.

---

## 📈 Model Performance

- Achieved **~97% accuracy**
- High precision and recall for spam detection

---

## 🧪 Sample Prediction

**Input:**
Congratulations! You have won a free lottery ticket worth $1000.
Click now to claim your prize.

**Output:**
Prediction: SPAM
Confidence: ~95%

---

## 🖥️ Run the Project Locally

###  Clone the repository
```bash
git clone https://github.com/your-username/spam-email-detector.git
cd spam-email-detector
python -m venv venv
venv\Scripts\activate
python train.py
python -m streamlit run app.py
