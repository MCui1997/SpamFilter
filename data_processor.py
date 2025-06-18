# data_processor.py - Data Loading and Preprocessing

"""
Data processing module for spam filter application.

This module handles dataset loading, text preprocessing, and data validation.
It provides functions to clean raw text messages, load and validate the spam dataset,
and calculate basic statistics for model training. The preprocessing pipeline
standardizes text format by removing noise (URLs, HTML, special characters) and
normalizing case to improve machine learning model performance.
"""

import pandas as pd
import re
import streamlit as st
from config import DATASET_FILE

"""
Text preprocessing function that cleans raw messages for machine learning.
Removes noise like URLs, HTML tags, and special characters while preserving
meaningful text content.
"""
def preprocess_text(text):
    """
    Clean and preprocess text for machine learning
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert everything to lowercase for consistency
    text = text.lower()
    
    # Remove all URLs (http, https, www) - they don't help classify spam vs ham
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags like <div>, <p> - focus on actual text content
    text = re.sub(r'<.*?>', '', text)
    
    # Remove numbers, punctuation, special chars - keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Clean up extra whitespace and normalize spacing
    text = ' '.join(text.split())
    
    return text

"""
Dataset loading function with validation and preprocessing.
Uses Streamlit caching to avoid reloading data on every interaction.
Validates dataset format and applies text cleaning to all messages.
"""
@st.cache_data
def load_dataset():
    """
    Load and preprocess the spam dataset
    
    Returns:
        pd.DataFrame: Processed dataset with 'label' and 'clean_message' columns
    """
    try:
        data = pd.read_csv(DATASET_FILE, sep='\t', header=None, names=['label', 'message'])
        
        if data.empty:
            raise ValueError("Dataset is empty")
        
        if 'label' not in data.columns or 'message' not in data.columns:
            raise ValueError("Dataset must have 'label' and 'message' columns")
        
        if not all(label in data['label'].values for label in ['spam', 'ham']):
            raise ValueError("Dataset must contain both 'spam' and 'ham' labels")
        
        data['clean_message'] = data['message'].apply(preprocess_text)
        data = data[data['clean_message'].str.len() > 0]
        
        return data
        
    except FileNotFoundError:
        st.error(f"❌ Dataset file '{DATASET_FILE}' not found!")
        st.info("Please ensure your dataset file is in the same folder as this app.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

"""
Statistical analysis function for dataset summary.
Calculates distribution of spam vs ham messages for model evaluation
and user interface display.
"""
def get_dataset_stats(data):
    """
    Calculate basic statistics about the dataset
    
    Args:
        data (pd.DataFrame): The dataset
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    if data is None:
        return None
        
    total_messages = len(data)
    spam_count = len(data[data['label'] == 'spam'])
    ham_count = len(data[data['label'] == 'ham'])
    spam_percentage = (spam_count / total_messages) * 100
    
    return {
        'total_messages': total_messages,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'spam_percentage': spam_percentage,
        'ham_percentage': 100 - spam_percentage
    }