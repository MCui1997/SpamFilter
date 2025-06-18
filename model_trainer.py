# model_trainer.py - Model Training and Evaluation

"""
Machine learning module for spam filter training and classification.

This module implements the core Naive Bayes spam detection algorithm using TF-IDF
feature extraction. It handles model training, evaluation, and real-time message
classification. The module also extracts dynamically learned spam keywords to
demonstrate the model's automatic pattern discovery capabilities, fulfilling
the assignment requirement for non-hardcoded spam detection.
"""

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from config import MODEL_CONFIG, TFIDF_CONFIG

"""
Main training function that implements the complete machine learning pipeline.
Uses Streamlit caching to avoid retraining on every interaction.
Returns trained model, vectorizer, and performance metrics.
"""
@st.cache_resource
def train_spam_filter(data):
    """
    Train the Naive Bayes spam filter
    
    Args:
        data (pd.DataFrame): Preprocessed dataset
        
    Returns:
        tuple: (model, vectorizer, accuracy, classification_report, spam_keywords)
    """
    if data is None or data.empty:
        return None, None, 0, "", []
    
    try:
        X = data['clean_message']
        y = data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        model = MultinomialNB(alpha=MODEL_CONFIG['naive_bayes_alpha'])
        model.fit(X_train_tfidf, y_train)
        
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        spam_keywords = extract_spam_keywords(vectorizer, model)
        
        return model, vectorizer, accuracy, class_report, spam_keywords
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None, 0, "", []

"""
Dynamic keyword extraction function that demonstrates automatic pattern learning.
Analyzes the trained model to identify words most predictive of spam without
hardcoded rules, fulfilling the key assignment requirement.
"""
def extract_spam_keywords(vectorizer, model, n=15):
    """
    Extract the words the model learned are most predictive of spam
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained Naive Bayes model
        n (int): Number of top keywords to return
        
    Returns:
        list: Top spam keywords discovered by the model
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
        spam_log_probs = model.feature_log_prob_[1, :]
        top_spam_indices = spam_log_probs.argsort()[-n:]
        spam_keywords = [feature_names[i] for i in top_spam_indices]
        
        return list(reversed(spam_keywords))
        
    except Exception as e:
        st.error(f"Error extracting keywords: {str(e)}")
        return []

"""
Real-time message classification function for user interface.
Applies the same preprocessing pipeline used during training to ensure
consistent feature extraction for new messages.
"""
def classify_message(message, model, vectorizer):
    """
    Classify a single message as spam or ham
    
    Args:
        message (str): Message to classify
        model: Trained model
        vectorizer: Fitted vectorizer
        
    Returns:
        tuple: (prediction, confidence)
    """
    if not message.strip():
        return None, 0
    
    try:
        from data_processor import preprocess_text
        
        clean_msg = preprocess_text(message)
        
        if not clean_msg.strip():
            return None, 0
        
        msg_tfidf = vectorizer.transform([clean_msg])
        prediction = model.predict(msg_tfidf)[0]
        probabilities = model.predict_proba(msg_tfidf)[0]
        confidence = max(probabilities) * 100
        
        return prediction, confidence
        
    except Exception as e:
        st.error(f"Error classifying message: {str(e)}")
        return None, 0