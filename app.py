# app.py - Main Streamlit Application
"""
AI-Powered Spam Detection Application

This Streamlit application provides a web interface for:
- Uploading and validating spam detection datasets
- Training machine learning models using Naive Bayes
- Classifying messages as spam or legitimate
- Displaying model performance metrics and insights

Dependencies:
    - streamlit: Web application framework
    - pandas: Data manipulation
    - Custom modules: config, data_processor, model_trainer
"""

import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="AI Spam Filter",
    layout="wide",
    page_icon="🛡️"
)

from io import StringIO
import pandas as pd

from data_processor import preprocess_text, get_dataset_stats
from model_trainer import train_spam_filter, classify_message

# Constants
PREVIEW_LINES = 5
LINE_TRUNCATE_LENGTH = 80
TOP_KEYWORDS_DISPLAY = 8
SESSION_STATE_KEYS = [
    'model_trained', 'model', 'vectorizer', 'accuracy', 
    'class_report', 'spam_keywords', 'stats'
]


def load_css(file_name: str) -> None:
    """
    Load CSS from external file for custom styling.
    
    Args:
        file_name (str): Path to the CSS file
    """
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styling.")


def clear_session_state() -> None:
    """Clear all model-related session state variables."""
    for key in SESSION_STATE_KEYS:
        if key in st.session_state:
            del st.session_state[key]


def preview_uploaded_file(uploaded_file) -> None:
    """
    Display a preview of the uploaded file content.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    """
    with st.expander("Preview file"):
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            preview_lines = stringio.getvalue().split('\n')[:PREVIEW_LINES]
            
            for i, line in enumerate(preview_lines, 1):
                if line.strip():
                    display_line = line[:LINE_TRUNCATE_LENGTH]
                    if len(line) > LINE_TRUNCATE_LENGTH:
                        display_line += "..."
                    st.text(f"Line {i}: {display_line}")
                    
        except UnicodeDecodeError:
            st.error("Cannot decode file. Please ensure it's a valid text file.")
        except Exception as e:
            st.error(f"Cannot preview file: {str(e)}")


def process_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """
    Process and validate the uploaded dataset file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame: Processed dataset or None if validation fails
    """
    try:
        # Read the uploaded file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(stringio, sep='\t', header=None, names=['label', 'message'])
        
        # Validation checks
        if data.empty:
            st.error("❌ Dataset is empty")
            return None
        
        if len(data.columns) != 2:
            st.error("❌ Dataset must have exactly 2 columns (label and message)")
            return None
        
        # Check for required labels
        required_labels = {'spam', 'ham'}
        available_labels = set(data['label'].unique())
        
        if not required_labels.issubset(available_labels):
            missing_labels = required_labels - available_labels
            st.error(f"❌ Dataset missing required labels: {', '.join(missing_labels)}")
            return None
        
        # Preprocess messages
        data['clean_message'] = data['message'].apply(preprocess_text)
        
        # Remove empty messages after preprocessing
        data = data[data['clean_message'].str.len() > 0]
        
        if data.empty:
            st.error("❌ No valid messages found after preprocessing")
            return None
        
        return data
        
    except UnicodeDecodeError:
        st.error("❌ Cannot decode file. Please ensure it's a valid UTF-8 encoded text file.")
        return None
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file appears to be empty.")
        return None
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")
        return None


def render_dataset_format_info() -> None:
    """Display information about the required dataset format."""
    st.header("📋 Dataset Format")
    st.markdown("""
    Your dataset should be a **tab-separated** file with this format:
    
    ```
    ham	Hi, how are you?
    spam	FREE! Win $1000 now!
    ham	Meeting at 3pm today
    spam	URGENT! Click this link
    ```
    
    """)


def render_upload_interface() -> None:
    """Render the dataset upload and training interface."""
    st.header("📤 Upload Dataset")
    
    uploaded_dataset = st.file_uploader(
        "Choose your dataset file",
        type=['txt', 'tsv'],
        help="Upload a tab-separated file with format: label \\t message"
    )
    
    if uploaded_dataset is not None:
        preview_uploaded_file(uploaded_dataset)
        
        # Train button
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model..."):
                data = process_uploaded_dataset(uploaded_dataset)
                
                if data is not None:
                    stats = get_dataset_stats(data)
                    model, vectorizer, accuracy, class_report, spam_keywords = train_spam_filter(data)
                    
                    if model is not None:
                        # Store in session state
                        st.session_state.update({
                            'model': model,
                            'vectorizer': vectorizer,
                            'accuracy': accuracy,
                            'class_report': class_report,
                            'spam_keywords': spam_keywords,
                            'stats': stats,
                            'model_trained': True
                        })
                        
                        st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.1f}%")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Failed to train model")


def render_message_classification() -> None:
    """Render the message classification interface."""
    st.header("🔍 Test Message")
    
    input_method = st.radio(
        "Input method:",
        ["Type Message", "Upload File"],
        horizontal=True
    )
    
    message_to_classify = ""
    
    if input_method == "Type Message":
        message_to_classify = st.text_area(
            "Enter your message:",
            height=150,
            placeholder="Type your email or SMS message here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload message file",
            type=['txt'],
            help="Upload a .txt file containing the message to classify"
        )
        
        if uploaded_file is not None:
            try:
                message_to_classify = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                st.text_area("File content:", message_to_classify, height=100, disabled=True)
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Classification
    if st.button("🎯 Classify Message", type="primary", use_container_width=True):
        if message_to_classify.strip():
            with st.spinner("Analyzing..."):
                prediction, confidence = classify_message(
                    message_to_classify, 
                    st.session_state.model, 
                    st.session_state.vectorizer
                )
                
                if prediction:
                    if prediction == "spam":
                        st.error(f"🚨 **SPAM DETECTED** (Confidence: {confidence:.1f}%)")
                    else:
                        st.success(f"✅ **LEGITIMATE MESSAGE** (Confidence: {confidence:.1f}%)")
        else:
            st.warning("⚠️ Please enter a message to classify")


def render_model_statistics() -> None:
    """Display model performance statistics."""
    st.header("📊 Model Statistics")
    
    accuracy = st.session_state.accuracy
    stats = st.session_state.stats
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.1f}%")
    with col2:
        st.metric("Total Messages", f"{stats['total_messages']:,}")
    with col3:
        st.metric("Spam Messages", f"{stats['spam_count']:,}")
    with col4:
        st.metric("Legitimate Messages", f"{stats['ham_count']:,}")


def render_spam_keywords() -> None:
    """Display top spam keywords identified by the model."""
    st.header("🔑 Spam Keywords")
    st.write("Top spam indicators discovered by the AI:")
    
    spam_keywords = st.session_state.spam_keywords
    
    # Display in columns for better layout
    col1, col2 = st.columns(2)
    
    for i, keyword in enumerate(spam_keywords[:TOP_KEYWORDS_DISPLAY], 1):
        target_col = col1 if i <= TOP_KEYWORDS_DISPLAY // 2 else col2
        target_col.write(f"**{i}.** {keyword}")


def main() -> None:
    """Main application function."""
    # Load external CSS
    load_css('styles.css')
    
    # Header
    st.title("🛡️ AI Spam Filter")
    st.subheader("Machine Learning Spam Detection using Naive Bayes")
    st.divider()
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    if not st.session_state.model_trained:
        # Upload and training interface
        render_upload_interface()
        st.divider()
        render_dataset_format_info()
        
    else:
        # Model is trained - show application interface
        accuracy = st.session_state.accuracy
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.1f}%")
        
        if st.button("📤 Upload New Dataset"):
            clear_session_state()
            st.rerun()
        
        st.divider()
        
        # Main application sections
        render_message_classification()
        st.divider()
        render_model_statistics()
        st.divider()
        render_spam_keywords()


if __name__ == "__main__":
    main()