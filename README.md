# Hate Speech Classification using Simple RNN

A binary text classification project that detects hate speech in tweets using a **Simple Recurrent Neural Network (RNN)**.

The project includes text preprocessing, vocabulary creation, sequence padding, Simple RNN model training, evaluation, and a Streamlit web application for real-time prediction.

---

## 🚀 Live Demo

**Streamlit App:**  
https://your-app-name.streamlit.app

> Replace the above URL with your actual Streamlit deployment URL after deployment.

---

## 📌 Problem Statement

Hate speech on social media platforms can negatively affect online communities. Automatically identifying potentially hateful content can help in content moderation and analysis.

The objective of this project is to build a **binary classifier using a Simple RNN** that classifies tweets into:

- **0 — Non-Hate**
- **1 — Hate Speech**

The model is trained using tweet text and corresponding labels.

---

## 🎯 Project Objectives

- Load and preprocess tweet data.
- Clean tweet text using regular expressions.
- Tokenize tweets into words.
- Build a vocabulary containing the top 5,000 words.
- Use `<OOV>` for unknown words.
- Convert text into integer sequences.
- Pad sequences to a fixed length of 30.
- Perform an 80/20 stratified train-test split.
- Build a Simple RNN using TensorFlow/Keras.
- Train the model for 10 epochs.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
- Deploy the trained model using Streamlit.

---

## 📊 Dataset

The dataset contains tweets and their corresponding binary labels.

Only the following columns are used:

```text
tweet
label
