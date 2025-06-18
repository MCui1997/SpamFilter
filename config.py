# config.py - Configuration for Spam Filter

"""
Machine learning model parameters for training and evaluation.
Controls train/test split, random seed for reproducibility, and Naive Bayes smoothing.
"""
MODEL_CONFIG = {
    # 30% reserved for testing - ensures model works on unseen data, not just memorized training examples
    'test_size': 0.3,           
    
    # Fixed seed ensures same train/test split every time for reproducible results
    'random_state': 42,         
    
    # Laplace smoothing - adds small count to prevent zero probabilities when word never seen in a class
    'naive_bayes_alpha': 1.0    
}
"""
TF-IDF vectorization parameters for text feature extraction.
Defines vocabulary size, n-gram range, and frequency thresholds.
"""
TFIDF_CONFIG = {
    'max_features': 5000,
    'stop_words': 'english',
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95
}

"""
Streamlit user interface configuration.
Sets page title, icon, and layout style.
"""
UI_CONFIG = {
    'page_title': "AI Spam Filter",
    'page_icon': "🛡️",
    'layout': "wide"
}

"""
Dataset file location.
Update this to match your actual dataset filename.
"""
DATASET_FILE = "spam_dataset.txt"

"""
Predefined test messages for demonstration purposes.
Covers different message types to showcase classification capabilities.
"""
TEST_EXAMPLES = {
    "Obvious Spam": "Congratulations! You've won $5000! Click here NOW to claim your FREE prize! Limited time offer!",
    "Normal Email": "Hi John, can we reschedule our meeting to 3 PM tomorrow? Let me know if that works for you.",
    "Financial Spam": "URGENT! Your account will be closed unless you verify immediately. Click this link and enter your password.",
    "Casual Message": "Hey mom, I'll be home late tonight. Don't wait up for dinner. Love you!",
    "Promotional": "Limited time offer! Get 50% off your next purchase. Use code SAVE50 at checkout.",
    "Work Email": "Please review the attached quarterly report and send feedback by Friday. Thanks!"
}