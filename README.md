Deep Learning for Comment Toxicity Detection
📌 Project Overview

This project focuses on detecting toxic comments using deep learning techniques. It analyzes user input text and predicts whether the comment is toxic or safe, along with probabilities for multiple toxicity categories.

🎯 Objective

To build an automated system that identifies toxic comments such as:

Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate

This helps improve online community moderation and user safety.

🧠 Technologies Used
Python
TensorFlow / Keras
NLP (Text Preprocessing)
Streamlit
Pandas, NumPy
Matplotlib, Seaborn
📂 Dataset
Source: Jigsaw Toxic Comment Dataset
Contains user comments with multiple toxicity labels
⚙️ Project Workflow
1. Data Preprocessing
Text cleaning (lowercase, remove symbols, URLs)
Stopword removal
Tokenization
Padding sequences
2. Model Development
CNN model for pattern detection
LSTM model for context understanding
Model training with early stopping
3. Model Evaluation
ROC-AUC Score
Classification Report
Confusion Matrix
4. Model Selection
Compared CNN and LSTM
Selected best model based on AUC score
5. Deployment
Built an interactive Streamlit web app
Real-time prediction of user input
🚀 Features
Real-time toxicity detection
Multi-label classification
Clean and user-friendly UI
Data insights visualization
🖥️ Streamlit App
Pages:
Home – Project overview
Data Insights – Label distribution visualization
Prediction – Real-time comment analysis
About – Project summary
▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run the app
streamlit run comment.py
📊 Sample Output
Displays:
Toxic / Non-Toxic classification
Probability for each label
🎯 Business Use Cases
Social media moderation
Online forums
E-learning platforms
Content filtering systems
🔍 Conclusion

The project successfully detects toxic comments using deep learning models and provides real-time predictions through a user-friendly interface. It can be used to improve online safety and automate content moderation.

👩‍💻 Author

Kaviya V
