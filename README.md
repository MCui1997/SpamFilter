# AI Spam Filter

A machine learning-powered spam detection application built with Streamlit and scikit-learn. This project demonstrates automated spam classification using Naive Bayes algorithm with TF-IDF feature extraction.

<a href="https://spamfilter.streamlit.app/" target="_blank">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
</a>

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages: `streamlit pandas scikit-learn`

### Installation & Running
```bash
pip install streamlit pandas scikit-learn
streamlit run app.py
```

Access the application at `http://localhost:8501`

## Usage

### Dataset Format
Create a tab-separated file with this format:
```
ham	Hi, how are you doing today?
spam	FREE! Win $1000 now! Click here immediately!
ham	Meeting scheduled for 3pm tomorrow
spam	URGENT! Your account will be suspended
```

Requirements:
- Tab-separated values (not comma-separated)
- Two columns: `label` and `message`
- Labels must be exactly `'spam'` or `'ham'`

### Workflow
1. Upload your dataset file (.txt or .tsv)
2. Click "Train Model" and wait for completion
3. Test messages using the classification interface
4. Review model statistics and learned spam keywords

## Project Architecture

### File Structure
```
├── app.py              # Main Streamlit application
├── config.py           # Configuration parameters
├── data_processor.py   # Data loading and preprocessing
├── model_trainer.py    # Machine learning pipeline
└── styles.css          # Custom styling (optional)
```

### Component Overview

**app.py** - Main application interface
- Handles file uploads and dataset validation
- Provides training and classification interfaces
- Displays model performance metrics

**config.py** - Centralized configuration
- Model parameters (test split, random seed, smoothing)
- TF-IDF settings (vocabulary size, n-grams, frequency thresholds)
- UI configuration and test examples

**data_processor.py** - Data preprocessing
- Text cleaning (removes URLs, HTML, special characters)
- Dataset validation and statistics calculation
- Preprocessing pipeline for consistent text format

**model_trainer.py** - Machine learning pipeline
- Model training with train/test split
- Dynamic spam keyword extraction
- Real-time message classification

## Mathematical Foundation & Theory

### Naive Bayes Classification

The core algorithm uses **Multinomial Naive Bayes**, which applies Bayes' theorem with the "naive" assumption of feature independence:

```
P(spam|message) = P(message|spam) × P(spam) / P(message)
```

Where:
- `P(spam|message)` = Probability message is spam given its content
- `P(message|spam)` = Likelihood of message content given it's spam
- `P(spam)` = Prior probability of spam (class distribution)
- `P(message)` = Evidence (normalizing constant)

**Multinomial Assumption**: Each word count follows a multinomial distribution, making it ideal for text classification where features represent word frequencies.

**Laplace Smoothing**: Applied with α=1.0 to handle unseen words:
```
P(word|class) = (count(word, class) + α) / (total_words_in_class + α × vocabulary_size)
```

### TF-IDF Feature Extraction

**Term Frequency (TF)**: Measures word importance within a document
```
TF(word, doc) = count(word, doc) / total_words(doc)
```

**Inverse Document Frequency (IDF)**: Reduces weight of common words
```
IDF(word) = log(total_documents / documents_containing(word))
```

**Combined TF-IDF Score**:
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)
```

### Model Parameters

**Training Configuration**:
- Train/test split: 70/30 with stratification
- Random state: 42 (reproducibility)
- Naive Bayes alpha: 1.0 (Laplace smoothing)

**TF-IDF Configuration**:
- Max features: 5,000 (vocabulary limit)
- N-gram range: (1,2) (unigrams and bigrams)
- Min document frequency: 2 (ignore rare terms)
- Max document frequency: 0.95 (ignore very common terms)

### Feature Selection Logic

**N-grams**: Captures both individual words and word pairs
- Unigrams: "free", "money", "click"
- Bigrams: "free money", "click here", "limited time"

**Frequency Filtering**:
- `min_df=2`: Eliminates noise from typos and rare words
- `max_df=0.95`: Removes uninformative common words beyond stop words

### Dynamic Keyword Extraction

The system identifies spam indicators by analyzing learned model weights:

```python
# Extract features with highest spam probability
spam_log_probs = model.feature_log_prob_[1, :]  # Log probabilities for spam class
top_indices = spam_log_probs.argsort()[-n:]     # Highest probability features
spam_keywords = [feature_names[i] for i in top_indices]
```

This demonstrates genuine machine learning by discovering patterns from data rather than using hardcoded rules.

## Performance & Evaluation

**Metrics Used**:
- Accuracy: Overall classification correctness
- Classification Report: Precision, recall, F1-score per class
- Confidence Scores: Maximum predicted probability (0-100%)

**Expected Performance**: 95-98% accuracy on balanced datasets with sufficient training data

**Model Interpretability**: Top spam keywords show what the algorithm learned (e.g., "free", "urgent", "click now")

## Technical Implementation Details

**Text Preprocessing Pipeline**:
1. Convert to lowercase
2. Remove URLs and HTML tags
3. Remove special characters and numbers
4. Normalize whitespace

**Training Process**:
1. Split data with stratification
2. Fit TF-IDF vectorizer on training set
3. Transform text to numerical vectors
4. Train Naive Bayes classifier
5. Evaluate on held-out test set

**Classification Process**:
1. Apply same preprocessing as training
2. Transform using fitted vectorizer
3. Predict class and probability scores
4. Return prediction with confidence

## Configuration Parameters

Key parameters in `config.py`:

```python
MODEL_CONFIG = {
    'test_size': 0.3,           # 30% for testing
    'random_state': 42,         # Reproducibility
    'naive_bayes_alpha': 1.0    # Laplace smoothing
}

TFIDF_CONFIG = {
    'max_features': 5000,       # Vocabulary size
    'ngram_range': (1, 2),      # Unigrams + bigrams
    'min_df': 2,                # Minimum document frequency
    'max_df': 0.95              # Maximum document frequency
}
```

## Troubleshooting

**Common Issues**:
- Ensure dataset is tab-separated (not comma or space)
- Check labels are exactly 'spam' and 'ham' (case-sensitive)
- Verify UTF-8 encoding for special characters
- Use balanced datasets (similar numbers of spam and ham messages)