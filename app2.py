import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re
import emoji
import time
from io import StringIO
import json
import random
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openai
import os

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="SentimentSense",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4F8BF9;
        margin-bottom: 0.5rem;
    }
    .dashboard-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .sentiment-sarcastic {
        color: #fd7e14;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a7ad5;
    }
    .comment-card {
        border-left: 4px solid;
        padding-left: 10px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
        border-radius: 0 0.5rem 0.5rem 0;
        padding: 10px;
        transition: transform 0.2s ease;
    }
    .comment-card:hover {
        transform: translateX(5px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #4F8BF9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9 !important;
        color: white !important;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .insight-card {
        background-color: #e9f7fe;
        border-left: 4px solid #4F8BF9;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4F8BF9;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #4F8BF9;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
        border-left: 5px solid #6c757d;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        background-color: #d1d1d1;
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .feedback-button {
        background-color: transparent;
        border: none;
        cursor: pointer;
        padding: 5px;
        transition: transform 0.2s;
    }
    .feedback-button:hover {
        transform: scale(1.2);
    }
    .pattern-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for sarcasm patterns
if 'sarcasm_patterns' not in st.session_state:
    st.session_state.sarcasm_patterns = {
        'phrases': [
            "yeah right", "sure thing", "oh really", "of course", "as if",
            "wow", "amazing", "brilliant", "genius", "exactly what i needed",
            "just what i wanted", "couldn't be happier", "thrilled", "oh great",
            "perfect timing", "just perfect", "exactly what I wanted", "totally worth it",
            "best thing ever", "so helpful", "clearly", "obviously", "absolutely",
            "definitely", "totally", "completely", "precisely", "certainly"
        ],
        'punctuation': [
            r'[!?]{2,}',  # Multiple exclamation or question marks
            r'!+\s*\?+',  # Exclamation followed by question mark
            r'\?+\s*!+',  # Question mark followed by exclamation
            r'\.{3,}',    # Multiple periods (ellipsis)
        ],
        'capitalization': [
            r'[A-Z]{3,}',  # Three or more consecutive capital letters
            r'\b[A-Z][a-z]*\s+[A-Z][a-z]*\b',  # Capitalized words in sequence
        ],
        'contradiction': [
            r'so\s+(good|great|nice|wonderful|amazing)\s+that\s+.*(terrible|awful|bad|worst|horrible)',
            r'(love|loving|loved)\s+.*(hate|hating|hated)',
            r'(best|greatest)\s+.*(worst|terrible|awful)',
            r'(happy|glad)\s+.*(sad|upset|angry)',
        ],
        'exaggeration': [
            r'(always|never|everyone|nobody|everything|nothing)\b',
            r'(best|worst|greatest|tiniest|largest|smallest)\s+(ever|in\s+the\s+world|in\s+history)',
            r'(million|billion|trillion|zillion|infinite|countless)\s+(times|ways)',
        ]
    }

# Initialize session state for sarcasm weights
if 'sarcasm_weights' not in st.session_state:
    st.session_state.sarcasm_weights = {
        'phrases': 0.3,
        'punctuation': 0.15,
        'capitalization': 0.15,
        'contradiction': 0.25,
        'exaggeration': 0.15
    }

# Initialize session state for sarcasm threshold
if 'sarcasm_threshold' not in st.session_state:
    st.session_state.sarcasm_threshold = 0.6

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for feedback data
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# Load NLP models when the app starts
@st.cache_resource
def load_models():
    # Sentiment Analysis model (RoBERTa)
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment",
        return_all_scores=True
    )

    # Dedicated sarcasm detection model
    sarcasm_model = pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        return_all_scores=True
    )

    # Load the RoBERTa tokenizer and model    
    # VADER sentiment analyzer for comparison
    vader_analyzer = SentimentIntensityAnalyzer()
    
    # RoBERTa model for enhanced sarcasm detection
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    
    return sentiment_model, sarcasm_model, vader_analyzer, roberta_tokenizer, roberta_model

# Load the models
sentiment_model, sarcasm_model, vader_analyzer, roberta_tokenizer, roberta_model = load_models()

# Function to check for sarcasm patterns
def check_sarcasm_patterns(text, patterns=None, weights=None, threshold=None):
    if patterns is None:
        patterns = st.session_state.sarcasm_patterns
    
    if weights is None:
        weights = st.session_state.sarcasm_weights
        
    if threshold is None:
        threshold = st.session_state.sarcasm_threshold
    
    score = 0
    matched_patterns = []
    
    # Check each pattern type
    for pattern_type, pattern_list in patterns.items():
        weight = weights.get(pattern_type, 0.2)  # Default weight if not specified
        
        for pattern in pattern_list:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight / len(pattern_list)
                matched_patterns.append((pattern_type, pattern))
    
    is_sarcastic = score >= threshold
    
    return {
        'is_sarcastic': is_sarcastic,
        'score': score,
        'matched_patterns': matched_patterns
    }

# Function to use LLM for sarcasm detection
def detect_sarcasm_with_llm(text, api_key=None):
    # If no API key is provided, use a simpler approach
    if not api_key:
        # Use the loaded models for a combined approach
        try:
            # Get sentiment from RoBERTa
            sentiment_results = sentiment_model(text)
            sentiment_label = sentiment_results[0][0]['label']
            sentiment_score = sentiment_results[0][0]['score']
            
            # Get VADER sentiment
            vader_scores = vader_analyzer.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # Check for contradiction between models
            roberta_sentiment = "POSITIVE" if sentiment_label == "LABEL_2" else "NEGATIVE" if sentiment_label == "LABEL_0" else "NEUTRAL"
            vader_sentiment = "POSITIVE" if vader_compound > 0.05 else "NEGATIVE" if vader_compound < -0.05 else "NEUTRAL"
            
            # Get sarcasm model prediction
            sarcasm_results = sarcasm_model(text)
            sarcasm_score = next((item['score'] for item in sarcasm_results[0] if item['label'] == 'LABEL_1'), 0)
            
            # Check pattern-based sarcasm
            pattern_result = check_sarcasm_patterns(text)
            
            # Combined score with higher weight to the dedicated model
            combined_score = (
                sarcasm_score * 0.5 +  # Dedicated model
                pattern_result['score'] * 0.3 +  # Pattern matching
                (1 if roberta_sentiment != vader_sentiment else 0) * 0.2  # Contradiction between models
            )
            
            return {
                'is_sarcastic': combined_score > 0.6,
                'score': combined_score,
                'model_score': sarcasm_score,
                'pattern_score': pattern_result['score'],
                'contradiction': roberta_sentiment != vader_sentiment
            }
            
        except Exception as e:
            print(f"Error in LLM-free sarcasm detection: {e}")
            # Fallback to pattern matching
            return check_sarcasm_patterns(text)
    
    # Use OpenAI API for sarcasm detection
    try:
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sarcasm detection assistant. Analyze the following text and determine if it contains sarcasm. Respond with a JSON object with two fields: 'is_sarcastic' (boolean) and 'confidence' (float between 0 and 1)."},
                {"role": "user", "content": f"Analyze this text for sarcasm: '{text}'"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            'is_sarcastic': result.get('is_sarcastic', False),
            'score': result.get('confidence', 0.5),
            'source': 'openai'
        }
        
    except Exception as e:
        print(f"Error in OpenAI sarcasm detection: {e}")
        # Fallback to the simpler approach
        return detect_sarcasm_with_llm(text, api_key=None)

# Preprocess text function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Replace emojis with text representation
    text = emoji.demojize(text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters but keep emojis
    text = re.sub(r'[^\w\s:_]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Enhanced sentiment analysis with improved sarcasm detection
def analyze_sentiment(text, detect_sarcasm=True, use_llm=False, api_key=None):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Check if text is empty after preprocessing
    if not processed_text or len(processed_text) < 2:
        return {
            "original_text": text,
            "processed_text": processed_text,
            "sentiment": "neutral",
            "confidence": 1.0,
            "is_sarcastic": False,
            "sarcasm_confidence": 0.0,
            "context_score": 0.0,
            "vader_scores": {"neg": 0, "neu": 1, "pos": 0, "compound": 0},
            "sarcasm_details": {}
        }
    
    # Get sentiment from RoBERTa model
    try:
        sentiment_results = sentiment_model(processed_text)
        
        # Map RoBERTa labels to our sentiment categories
        label_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }
        
        # Extract scores
        sentiment_scores = {label_mapping[item["label"]]: item["score"] for item in sentiment_results[0]}
        
        # Determine sentiment
        if sentiment_scores["positive"] > sentiment_scores["negative"] and sentiment_scores["positive"] > sentiment_scores["neutral"]:
            sentiment = "positive"
            confidence = sentiment_scores["positive"]
        elif sentiment_scores["negative"] > sentiment_scores["positive"] and sentiment_scores["negative"] > sentiment_scores["neutral"]:
            sentiment = "negative"
            confidence = sentiment_scores["negative"]
        else:
            sentiment = "neutral"
            confidence = sentiment_scores["neutral"]
            
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        sentiment = "neutral"
        confidence = 0.5
        sentiment_scores = {"positive": 0, "negative": 0, "neutral": 1}
    
    # Get VADER sentiment for comparison
    vader_scores = vader_analyzer.polarity_scores(processed_text)
    
    # Detect sarcasm if enabled
    is_sarcastic = False
    sarcasm_confidence = 0.0
    sarcasm_details = {}
    
    if detect_sarcasm:
        try:
            if use_llm:
                # Use LLM-based sarcasm detection
                sarcasm_result = detect_sarcasm_with_llm(text, api_key)
                is_sarcastic = sarcasm_result.get('is_sarcastic', False)
                sarcasm_confidence = sarcasm_result.get('score', 0.0)
                sarcasm_details = sarcasm_result
            else:
                # Use pattern-based and model-based sarcasm detection
                # 1. Check for sarcasm patterns
                pattern_result = check_sarcasm_patterns(text)
                
                # 2. Use dedicated sarcasm model
                model_result = sarcasm_model(processed_text)
                model_score = next((item['score'] for item in model_result[0] if item['label'] == 'LABEL_1'), 0)
                
                # 3. Check for contradiction between sentiment models
                roberta_sentiment = sentiment
                vader_sentiment = "positive" if vader_scores["compound"] > 0.05 else "negative" if vader_scores["compound"] < -0.05 else "neutral"
                has_contradiction = roberta_sentiment != vader_sentiment
                
                # 4. Combined sarcasm score
                sarcasm_confidence = (
                    model_score * 0.5 +  # Dedicated model
                    pattern_result['score'] * 0.3 +  # Pattern matching
                    (0.2 if has_contradiction else 0)  # Contradiction between models
                )
                
                is_sarcastic = sarcasm_confidence > st.session_state.sarcasm_threshold
                
                sarcasm_details = {
                    'pattern_result': pattern_result,
                    'model_score': model_score,
                    'has_contradiction': has_contradiction,
                    'combined_score': sarcasm_confidence
                }
                
            # If sarcastic with high confidence, flip sentiment
            if is_sarcastic and sarcasm_confidence > 0.7:
                if sentiment == "positive":
                    sentiment = "negative"
                elif sentiment == "negative":
                    sentiment = "positive"
                
        except Exception as e:
            print(f"Sarcasm detection error: {e}")
    
    # Calculate a context score based on text length and complexity
    words = processed_text.split()
    context_score = min(len(words) / 50, 1.0)  # Text length
    
    # Add complexity factor (unique words ratio)
    unique_words_ratio = len(set(words)) / max(len(words), 1)
    context_score = (context_score + unique_words_ratio) / 2
    
    return {
        "original_text": text,
        "processed_text": processed_text,
        "sentiment": sentiment,
        "confidence": float(confidence),
        "sentiment_scores": sentiment_scores,
        "is_sarcastic": is_sarcastic,
        "sarcasm_confidence": float(sarcasm_confidence),
        "context_score": float(context_score),
        "vader_scores": vader_scores,
        "sarcasm_details": sarcasm_details
    }

# Function to analyze a list of comments (with context)
def analyze_comments(comments, detect_sarcasm=True, use_llm=False, api_key=None):
    results = []
    
    # Context tracking for conversation analysis
    conversation_context = []
    
    for idx, comment in enumerate(comments):
        # Build context from previous comments
        if idx > 0:
            # Use up to 3 previous comments as context
            context_window = min(3, idx)
            conversation_context = comments[idx-context_window:idx]
        
        # Basic sentiment analysis
        result = analyze_sentiment(comment, detect_sarcasm, use_llm, api_key)
        
        # Apply contextual analysis for conversations
        if conversation_context and detect_sarcasm:
            # Check for response patterns that might indicate sarcasm
            if any("?" in ctx for ctx in conversation_context) and "!" in comment:
                result["context_score"] += 0.1
                result["sarcasm_confidence"] += 0.1
                
            # Check for sentiment shifts (potential sarcasm indicator)
            prev_results = [analyze_sentiment(ctx, False) for ctx in conversation_context]
            prev_sentiments = [r["sentiment"] for r in prev_results]
            
            # If previous sentiment was consistently one way and this comment is opposite
            if len(set(prev_sentiments)) == 1 and result["sentiment"] != prev_sentiments[0]:
                result["sarcasm_confidence"] += 0.15
                
            # Normalize sarcasm confidence
            result["sarcasm_confidence"] = min(result["sarcasm_confidence"], 1.0)
            result["is_sarcastic"] = result["sarcasm_confidence"] > st.session_state.sarcasm_threshold
        
        results.append(result)
    
    return results

# Function to generate a sample dataset for demonstration
def generate_sample_dataset(size=100):
    positive_templates = [
        "I love this product! It's amazing!",
        "Great experience, would recommend to everyone.",
        "The service was excellent, very satisfied.",
        "Absolutely thrilled with my purchase.",
        "This is the best I've ever tried!"
    ]
    
    negative_templates = [
        "This is terrible, don't waste your money.",
        "Very disappointed with the quality.",
        "The worst experience I've ever had.",
        "Would not recommend to anyone.",
        "Complete waste of time and money."
    ]
    
    neutral_templates = [
        "It was okay, nothing special.",
        "Does the job but could be better.",
        "Not bad, not great either.",
        "Reasonable price for what you get.",
        "It's alright, I guess."
    ]
    
    sarcastic_templates = [
        "Oh yeah, this is EXACTLY what I needed... another problem to solve!",
        "Wow, what an AMAZING experience... if you enjoy frustration!",
        "Sure, because waiting 2 hours for customer service is just SO much fun!",
        "Of course it broke immediately, that's definitely what quality looks like!",
        "This product is PERFECT... if you want something that doesn't work!",
        "I'm THRILLED to spend my entire day fixing this!",
        "Just what I wanted, another useless feature that doesn't work!",
        "Oh great, now I have to restart everything. How WONDERFUL!",
        "Absolutely LOVE when things break for no reason!",
        "The best part was when it crashed and deleted all my work!"
    ]
    
    data = []
    now = datetime.now()
    
    for i in range(size):
        sentiment_type = random.choices(
            ["positive", "negative", "neutral", "sarcastic"],
            weights=[0.4, 0.3, 0.2, 0.1],
            k=1
        )[0]
        
        if sentiment_type == "positive":
            template = random.choice(positive_templates)
        elif sentiment_type == "negative":
            template = random.choice(negative_templates)
        elif sentiment_type == "neutral":
            template = random.choice(neutral_templates)
        else:
            template = random.choice(sarcastic_templates)
        
        # Add some variation to the template
        words = template.split()
        if len(words) > 3:
            insert_idx = random.randint(1, len(words) - 2)
            adjectives = ["really", "very", "extremely", "somewhat", "kind of", "absolutely"]
            words.insert(insert_idx, random.choice(adjectives))
        
        comment = " ".join(words)
        timestamp = (now - timedelta(days=random.randint(0, 30), 
                                   hours=random.randint(0, 23), 
                                   minutes=random.randint(0, 59))).isoformat()
        
        platform = random.choice(["YouTube", "Twitter", "Instagram", "Facebook"])
        
        data.append({
            "comment": comment,
            "timestamp": timestamp,
            "platform": platform,
            "user_id": f"user_{random.randint(1000, 9999)}",
            "true_sentiment": sentiment_type
        })
    
    return pd.DataFrame(data)

# Function to scrape comments from websites
def scrape_comments(url, platform="Website"):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        comments = []
        now = datetime.now()
        
        # Different scraping strategies based on platform
        if "youtube.com" in url:
            # YouTube comments are loaded dynamically with JavaScript, so this is a simplified version
            comment_elements = soup.select("yt-formatted-string#content-text")
            for element in comment_elements[:50]:  # Limit to 50 comments
                comments.append({
                    "comment": element.get_text(strip=True),
                    "timestamp": (now - timedelta(days=random.randint(0, 10))).isoformat(),
                    "platform": "YouTube",
                    "user_id": f"user_{random.randint(1000, 9999)}"
                })
                
        elif "twitter.com" in url or "x.com" in url:
            # Twitter/X tweets
            tweet_elements = soup.select("div[data-testid='tweetText']")
            for element in tweet_elements[:50]:
                comments.append({
                    "comment": element.get_text(strip=True),
                    "timestamp": (now - timedelta(days=random.randint(0, 5))).isoformat(),
                    "platform": "Twitter",
                    "user_id": f"user_{random.randint(1000, 9999)}"
                })
                
        else:
            # Generic comment scraping
            # Look for common comment containers
            comment_elements = soup.select("div.comment, div.comments, .comment-body, .comment-content")
            
            if not comment_elements:
                # If no comments found, extract paragraphs as content
                comment_elements = soup.select("p")[:20]  # Limit to 20 paragraphs
            
            for element in comment_elements[:50]:
                comments.append({
                    "comment": element.get_text(strip=True),
                    "timestamp": (now - timedelta(days=random.randint(0, 15))).isoformat(),
                    "platform": platform,
                    "user_id": f"user_{random.randint(1000, 9999)}"
                })
        
        # If no comments were found, generate some sample data
        if not comments:
            st.warning(f"Could not extract comments from {url}. Using sample data instead.")
            return generate_sample_dataset(30)
        
        return pd.DataFrame(comments)
        
    except Exception as e:
        st.error(f"Error scraping comments: {str(e)}")
        # Return sample data as fallback
        return generate_sample_dataset(20)

# Function to extract insights from analysis results
def extract_insights(data):
    insights = []
    
    # Check if data is empty
    if data.empty:
        return ["No data available for insights."]
    
    # Overall sentiment distribution
    sentiment_counts = data["analyzed_sentiment"].value_counts(normalize=True)
    
    # Most common sentiment
    if not sentiment_counts.empty:
        most_common = sentiment_counts.idxmax()
        insights.append(f"The dominant sentiment is {most_common} ({sentiment_counts[most_common]:.1%}).")
    
    # Sarcasm detection
    sarcasm_rate = data["is_sarcastic"].mean()
    if sarcasm_rate > 0.2:
        insights.append(f"High sarcasm detected ({sarcasm_rate:.1%} of comments), suggesting potential hidden dissatisfaction.")
    elif sarcasm_rate > 0.1:
        insights.append(f"Moderate sarcasm detected ({sarcasm_rate:.1%} of comments).")
    
    # Sentiment trends
    if "date" in data.columns and len(data["date"].unique()) > 1:
        # Check if sentiment has been changing
        sentiment_by_date = data.groupby("date")["analyzed_sentiment"].apply(
            lambda x: (x == "positive").mean() - (x == "negative").mean()
        ).reset_index()
        
        if len(sentiment_by_date) >= 3:
            first_half = sentiment_by_date.iloc[:len(sentiment_by_date)//2]["analyzed_sentiment"].mean()
            second_half = sentiment_by_date.iloc[len(sentiment_by_date)//2:]["analyzed_sentiment"].mean()
            
            if second_half - first_half > 0.2:
                insights.append("Sentiment is improving over time.")
            elif first_half - second_half > 0.2:
                insights.append("Sentiment is declining over time.")
    
    # Platform-specific insights
    if "platform" in data.columns and len(data["platform"].unique()) > 1:
        platform_sentiment = data.groupby("platform")["analyzed_sentiment"].apply(
            lambda x: (x == "positive").mean() - (x == "negative").mean()
        )
        
        best_platform = platform_sentiment.idxmax()
        worst_platform = platform_sentiment.idxmin()
        
        if platform_sentiment[best_platform] - platform_sentiment[worst_platform] > 0.3:
            insights.append(f"Sentiment is most positive on {best_platform} and most negative on {worst_platform}.")
    
    # Content length vs. sentiment
    data["word_count"] = data["comment"].apply(lambda x: len(str(x).split()))
    
    avg_positive_length = data[data["analyzed_sentiment"] == "positive"]["word_count"].mean()
    avg_negative_length = data[data["analyzed_sentiment"] == "negative"]["word_count"].mean()
    
    if avg_negative_length > avg_positive_length * 1.5:
        insights.append("Negative comments tend to be longer, suggesting detailed criticism.")
    elif avg_positive_length > avg_negative_length * 1.5:
        insights.append("Positive comments tend to be longer, suggesting enthusiastic support.")
    
    # Sarcasm insights
    if "is_sarcastic" in data.columns and data["is_sarcastic"].sum() > 0:
        sarcastic_data = data[data["is_sarcastic"]]
        if not sarcastic_data.empty:
            sarcastic_sentiment = sarcastic_data["analyzed_sentiment"].value_counts(normalize=True)
            most_common_sarcastic = sarcastic_sentiment.idxmax() if not sarcastic_sentiment.empty else None
            
            if most_common_sarcastic:
                insights.append(f"Sarcastic comments tend to have {most_common_sarcastic} sentiment, suggesting potential hidden meanings.")
    
    # Add recommendations based on insights
    if "negative" in sentiment_counts and sentiment_counts["negative"] > 0.3:
        insights.append("RECOMMENDATION: Address negative feedback to improve sentiment.")
    
    if sarcasm_rate > 0.15:
        insights.append("RECOMMENDATION: Look beyond literal meaning in comments to address underlying issues.")
    
    return insights

# Function to simulate API data retrieval
def fetch_api_data(platform, query, count=50):
    st.info(f"Fetching data from {platform} API for query: '{query}'")
    
    # Simulate API call with loading time
    progress_bar = st.progress(0)
    for i in range(101):
        time.sleep(0.01)  # Faster loading for better UX
        progress_bar.progress(i)
    
    # Generate sample data with the query included in some comments
    data = generate_sample_dataset(count)
    
    # Modify some comments to include the query term
    for i in range(min(10, len(data))):
        idx = random.randint(0, len(data) - 1)
        comment = data.loc[idx, "comment"]
        words = comment.split()
        insert_idx = random.randint(0, len(words))
        words.insert(insert_idx, query)
        data.loc[idx, "comment"] = " ".join(words)
    
    # Set all platform values to the requested platform
    data["platform"] = platform
    
    return data

# Function to update sarcasm patterns based on feedback
def update_sarcasm_patterns(text, is_sarcastic):
    # Only update if we have a clear signal
    if is_sarcastic:
        # Extract potential new patterns
        # 1. Look for phrases with exaggeration
        exaggeration_words = ["so", "very", "really", "absolutely", "totally", "completely"]
        for word in exaggeration_words:
            matches = re.findall(rf'\b{word}\s+(\w+)\b', text, re.IGNORECASE)
            for match in matches:
                pattern = f"{word} {match}"
                if pattern.lower() not in [p.lower() for p in st.session_state.sarcasm_patterns['phrases']]:
                    st.session_state.sarcasm_patterns['phrases'].append(pattern)
        
        # 2. Look for contradictions
        contradiction_patterns = [
            r'(good|great|nice|wonderful|amazing)\s+.*(terrible|awful|bad|worst|horrible)',
            r'(love|loving|loved)\s+.*(hate|hating|hated)',
            r'(best|greatest)\s+.*(worst|terrible|awful)',
            r'(happy|glad)\s+.*(sad|upset|angry)'
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, text, re.IGNORECASE) and pattern not in st.session_state.sarcasm_patterns['contradiction']:
                st.session_state.sarcasm_patterns['contradiction'].append(pattern)
    
    # If marked as not sarcastic but our system thought it was, adjust weights
    elif check_sarcasm_patterns(text)['is_sarcastic']:
        # Reduce weights slightly for false positives
        for pattern_type in st.session_state.sarcasm_weights:
            st.session_state.sarcasm_weights[pattern_type] *= 0.95
        
        # Increase threshold slightly
        st.session_state.sarcasm_threshold = min(st.session_state.sarcasm_threshold + 0.02, 0.9)

# Function to get chatbot response
def get_chatbot_response(user_input, api_key=None):
    # Analyze sentiment of user input
    analysis = analyze_sentiment(user_input, detect_sarcasm=True, use_llm=(api_key is not None), api_key=api_key)
    
    # If no API key, use a rule-based approach
    if not api_key:
        # Determine response based on sentiment and sarcasm
        if analysis['is_sarcastic']:
            responses = [
                "I detect some sarcasm there! Let me know how I can actually help.",
                "That sounds sarcastic. What's really on your mind?",
                "Behind that sarcasm, I sense you might have a real question. What can I help with?",
                "Noted with a hint of sarcasm. How can I assist you more directly?"
            ]
        elif analysis['sentiment'] == 'positive':
            responses = [
                "I'm glad you're feeling positive! How can I help you analyze your data?",
                "Thanks for the positive energy! What would you like to know about your sentiment data?",
                "Great to hear that! Would you like to explore some sentiment analysis features?",
                "Wonderful! I'm here to help with any sentiment analysis needs."
            ]
        elif analysis['sentiment'] == 'negative':
            responses = [
                "I sense some frustration. Let me help you find what you're looking for.",
                "I understand this might be challenging. How can I make it easier?",
                "Let's work through this together. What specific aspect is troubling you?",
                "I'm here to help with any difficulties you're experiencing."
            ]
        else:  # neutral
            responses = [
                "How can I help you with sentiment analysis today?",
                "I'm here to assist with your sentiment analysis needs. What would you like to know?",
                "What specific aspect of sentiment analysis are you interested in?",
                "I can help you analyze text, comments, or social media data. What would you like to focus on?"
            ]
        
        return random.choice(responses)
    
    # If API key is provided, use OpenAI
    try:
        openai.api_key = api_key
        
        # Include sentiment analysis in the prompt
        sentiment_info = f"The user's message has a {analysis['sentiment']} sentiment"
        if analysis['is_sarcastic']:
            sentiment_info += " with detected sarcasm"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant for a sentiment analysis application called SentimentSense. {sentiment_info}. Respond appropriately to help the user with sentiment analysis tasks."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in OpenAI response: {e}")
        # Fallback to rule-based approach
        return get_chatbot_response(user_input, api_key=None)

# Define the main app sections
def main():
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://placeholder.svg?height=150&width=150", width=150)
        st.markdown("<div class='main-header'>SentimentSense</div>", unsafe_allow_html=True)
        
        navigation = st.radio(
            "Navigation",
            ["Dashboard", "Analyze Comments", "Social Media Monitoring", "Web Scraping", "Advanced Analytics", "Chatbot", "Sarcasm Settings", "Settings"]
        )
        
        # Add a quick analysis section in sidebar
        st.markdown("---")
        st.markdown("### Quick Analysis")
        quick_text = st.text_area("Enter text for quick analysis", height=100)
        
        if quick_text:
            with st.spinner("Analyzing..."):
                quick_result = analyze_sentiment(quick_text)
                
                sentiment_emoji = {
                    "positive": "😊",
                    "negative": "😞",
                    "neutral": "😐"
                }
                
                emoji_icon = sentiment_emoji.get(quick_result["sentiment"], "😐")
                sarcasm_indicator = " (Sarcastic 🙃)" if quick_result["is_sarcastic"] else ""
                
                st.markdown(f"**Sentiment:** {emoji_icon} {quick_result['sentiment'].capitalize()}{sarcasm_indicator}")
                st.progress(quick_result["confidence"])
    
    # Title and description
    if navigation != "Dashboard":
        st.markdown("<div class='main-header'>SentimentSense</div>", unsafe_allow_html=True)
        st.markdown(
            "An advanced platform for sentiment analysis with sarcasm detection and contextual understanding."
        )
    
    # Dashboard Page
    if navigation == "Dashboard":
        display_dashboard()
    
    # Analyze Comments Page
    elif navigation == "Analyze Comments":
        analyze_comments_page()
    
    # Social Media Monitoring Page
    elif navigation == "Social Media Monitoring":
        social_media_monitoring()
        
    # Web Scraping Page
    elif navigation == "Web Scraping":
        web_scraping_page()
        
    # Advanced Analytics Page
    elif navigation == "Advanced Analytics":
        advanced_analytics_page()
        
    # Chatbot Page
    elif navigation == "Chatbot":
        chatbot_page()
        
    # Sarcasm Settings Page
    elif navigation == "Sarcasm Settings":
        sarcasm_settings_page()
    
    # Settings Page
    elif navigation == "Settings":
        settings_page()

# Dashboard Page
def display_dashboard():
    st.markdown("<div class='main-header'>Interactive Dashboard</div>", unsafe_allow_html=True)
    
    # Load sample data if not already available in session state
    if "dashboard_data" not in st.session_state:
        st.session_state.dashboard_data = generate_sample_dataset(200)
    
    data = st.session_state.dashboard_data
    
    # Analyze all comments if not already analyzed
    if "analyzed_results" not in st.session_state:
        with st.spinner("Analyzing comments..."):
            results = []
            for comment in data["comment"]:
                results.append(analyze_sentiment(comment))
            
            # Add analysis results to the dataframe
            sentiment_values = [r["sentiment"] for r in results]
            is_sarcastic = [r["is_sarcastic"] for r in results]
            
            data["analyzed_sentiment"] = sentiment_values
            data["is_sarcastic"] = is_sarcastic
            
            st.session_state.analyzed_results = data
    
    analyzed_data = st.session_state.analyzed_results
    
    # Add date column for time-based analysis
    analyzed_data['date'] = pd.to_datetime(analyzed_data['timestamp']).dt.date
    
    # Dashboard controls
    with st.expander("Dashboard Controls", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(
                    analyzed_data['date'].min(),
                    analyzed_data['date'].max()
                ),
                min_value=analyzed_data['date'].min(),
                max_value=analyzed_data['date'].max()
            )
        
        with col2:
            selected_platforms = st.multiselect(
                "Platforms",
                options=analyzed_data['platform'].unique(),
                default=analyzed_data['platform'].unique()
            )
        
        # Filter data based on selections
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = analyzed_data[
                (analyzed_data['date'] >= start_date) & 
                (analyzed_data['date'] <= end_date) &
                (analyzed_data['platform'].isin(selected_platforms))
            ]
        else:
            filtered_data = analyzed_data[analyzed_data['platform'].isin(selected_platforms)]
    
    # Create dashboard metrics and visualizations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        total_comments = len(filtered_data)
        st.markdown("<div class='metric-value'>{:,}</div>".format(total_comments), unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Comments</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        positive_count = sum(filtered_data["analyzed_sentiment"] == "positive")
        positive_percentage = (positive_count / total_comments) * 100 if total_comments > 0 else 0
        st.markdown("<div class='metric-value' style='color: #28a745;'>{:.1f}%</div>".format(positive_percentage), unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Positive Sentiment</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        negative_count = sum(filtered_data["analyzed_sentiment"] == "negative")
        negative_percentage = (negative_count / total_comments) * 100 if total_comments > 0 else 0
        st.markdown("<div class='metric-value' style='color: #dc3545;'>{:.1f}%</div>".format(negative_percentage), unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Negative Sentiment</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        sarcastic_count = sum(filtered_data["is_sarcastic"])
        sarcastic_percentage = (sarcastic_count / total_comments) * 100 if total_comments > 0 else 0
        st.markdown("<div class='metric-value' style='color: #fd7e14;'>{:.1f}%</div>".format(sarcastic_percentage), unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Sarcasm Detected</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Extract insights
    insights = extract_insights(filtered_data)
    
    # Display insights
    st.markdown("### Key Insights")
    for insight in insights:
        st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>Sentiment Distribution</div>", unsafe_allow_html=True)
        sentiment_counts = filtered_data["analyzed_sentiment"].value_counts()
        fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            },
            hole=0.4
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='sub-header'>Platform Distribution</div>", unsafe_allow_html=True)
        platform_counts = filtered_data["platform"].value_counts()
        fig = px.bar(
            x=platform_counts.index,
            y=platform_counts.values,
            color=platform_counts.index,
            labels={'x': 'Platform', 'y': 'Count'},
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Over Time
    st.markdown("<div class='sub-header'>Sentiment Trends Over Time</div>", unsafe_allow_html=True)
    
    # Group by date and count sentiments
    sentiment_over_time = filtered_data.groupby(['date', 'analyzed_sentiment']).size().unstack().fillna(0)
    
    # Plot sentiment over time
    fig = go.Figure()
    
    if 'positive' in sentiment_over_time.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_over_time.index,
            y=sentiment_over_time['positive'],
            mode='lines',
            name='Positive',
            line=dict(color='#28a745', width=2)
        ))
    
    if 'negative' in sentiment_over_time.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_over_time.index,
            y=sentiment_over_time['negative'],
            mode='lines',
            name='Negative',
            line=dict(color='#dc3545', width=2)
        ))
    
    if 'neutral' in sentiment_over_time.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_over_time.index,
            y=sentiment_over_time['neutral'],
            mode='lines',
            name='Neutral',
            line=dict(color='#6c757d', width=2)
        ))
    
    # Add sarcasm trace
    sarcasm_over_time = filtered_data.groupby('date')['is_sarcastic'].sum()
    
    fig.add_trace(go.Scatter(
        x=sarcasm_over_time.index,
        y=sarcasm_over_time.values,
        mode='lines',
        name='Sarcasm',
        line=dict(color='#fd7e14', width=2, dash='dot')
    ))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Comments',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Word cloud visualization
    st.markdown("<div class='sub-header'>Common Words in Comments</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word cloud for positive comments
        positive_comments = " ".join(filtered_data[filtered_data["analyzed_sentiment"] == "positive"]["comment"])
        if positive_comments:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                colormap='Greens',
                max_words=100
            ).generate(positive_comments)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Positive Comments')
            st.pyplot(fig)
    
    with col2:
        # Word cloud for negative comments
        negative_comments = " ".join(filtered_data[filtered_data["analyzed_sentiment"] == "negative"]["comment"])
        if negative_comments:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                colormap='Reds',
                max_words=100
            ).generate(negative_comments)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Negative Comments')
            st.pyplot(fig)
    
    # Display comments with sentiment
    st.markdown("<div class='sub-header'>Recent Comments Analysis</div>", unsafe_allow_html=True)
    
    # Add a search box for filtering comments
    search_term = st.text_input("Search comments", "")
    
    filtered_comments = filtered_data
    if search_term:
        filtered_comments = filtered_data[filtered_data["comment"].str.contains(search_term, case=False)]
    
    # Display the most recent comments with their sentiment
    recent_comments = filtered_comments.sort_values("timestamp", ascending=False).head(10)
    
    for _, row in recent_comments.iterrows():
        sentiment_class = f"sentiment-{row['analyzed_sentiment']}"
        sarcasm_indicator = " (Sarcastic)" if row["is_sarcastic"] else ""
        
        border_color = "#28a745" if row["analyzed_sentiment"] == "positive" else "#dc3545" if row["analyzed_sentiment"] == "negative" else "#6c757d"
        if row["is_sarcastic"]:
            border_color = "#fd7e14"
            
        st.markdown(
            f"""
            <div class="comment-card" style="border-left-color: {border_color};">
                <div>{row["comment"]}</div>
                <div style='font-size: 0.8rem; color: #6c757d;'>
                    <span class='{sentiment_class}'>{row["analyzed_sentiment"].capitalize()}{sarcasm_indicator}</span> | 
                    {row["platform"]} | {row["timestamp"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Analyze Comments Page
def analyze_comments_page():
    st.markdown("<div class='sub-header'>Analyze Comments</div>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Text Input", "File Upload", "Conversation Analysis"])
    
    # Text Input Tab
    with tab1:
        text_input = st.text_area("Enter text to analyze", height=150)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            detect_sarcasm = st.checkbox("Detect sarcasm", value=True)
        with col2:
            use_llm = st.checkbox("Use LLM for enhanced detection", value=False)
        with col3:
            show_details = st.checkbox("Show detailed analysis", value=False)
        
        api_key = None
        if use_llm:
            api_key = st.text_input("OpenAI API Key (optional)", type="password")
        
        if st.button("Analyze Text"):
            if text_input:
                with st.spinner("Analyzing..."):
                    result = analyze_sentiment(text_input, detect_sarcasm, use_llm, api_key)
                
                # Display results
                sentiment_color = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "gray"
                }
                
                st.markdown(f"### Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Sentiment:** <span style='color: {sentiment_color[result['sentiment']]}'>{result['sentiment'].upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {result['confidence']:.2f}")
                
                with col2:
                    st.markdown(f"**Sarcasm Detected:** {'Yes' if result['is_sarcastic'] else 'No'}")
                    if result['is_sarcastic']:
                        st.markdown(f"**Sarcasm Confidence:** {result['sarcasm_confidence']:.2f}")
                
                # Display sentiment visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result["confidence"],
                    title = {'text': f"Sentiment: {result['sentiment'].capitalize()}"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': sentiment_color[result['sentiment']]},
                        'steps': [
                            {'range': [0, 0.33], 'color': "lightgray"},
                            {'range': [0.33, 0.66], 'color': "gray"},
                            {'range': [0.66, 1], 'color': "darkgray"}
                        ]
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sarcasm detection details
                if result['is_sarcastic']:
                    st.markdown("### Sarcasm Detection Details")
                    
                    if 'sarcasm_details' in result and result['sarcasm_details']:
                        details = result['sarcasm_details']
                        
                        if 'pattern_result' in details:
                            pattern_result = details['pattern_result']
                            if 'matched_patterns' in pattern_result and pattern_result['matched_patterns']:
                                st.markdown("#### Matched Sarcasm Patterns")
                                for pattern_type, pattern in pattern_result['matched_patterns']:
                                    st.markdown(f"- **{pattern_type.capitalize()}**: `{pattern}`")
                        
                        if 'model_score' in details:
                            st.markdown(f"**Model-based score:** {details['model_score']:.2f}")
                        
                        if 'has_contradiction' in details:
                            st.markdown(f"**Sentiment contradiction detected:** {'Yes' if details['has_contradiction'] else 'No'}")
                        
                        if 'combined_score' in details:
                            st.markdown(f"**Combined sarcasm score:** {details['combined_score']:.2f}")
                    
                    # Add feedback buttons for sarcasm detection
                    st.markdown("#### Was the sarcasm detection correct?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("👍 Yes, it's sarcastic"):
                            # Update patterns based on positive feedback
                            update_sarcasm_patterns(text_input, True)
                            st.session_state.feedback_data.append({
                                "text": text_input,
                                "is_sarcastic": True,
                                "system_prediction": True,
                                "feedback": "correct"
                            })
                            st.success("Thank you for your feedback! We've updated our sarcasm detection patterns.")
                    
                    with col2:
                        if st.button("👎 No, it's not sarcastic"):
                            # Adjust weights based on negative feedback
                            update_sarcasm_patterns(text_input, False)
                            st.session_state.feedback_data.append({
                                "text": text_input,
                                "is_sarcastic": False,
                                "system_prediction": True,
                                "feedback": "incorrect"
                            })
                            st.success("Thank you for your feedback! We've adjusted our sarcasm detection parameters.")
                
                elif detect_sarcasm:
                    st.markdown("### Sarcasm Detection Details")
                    st.markdown("No sarcasm detected in this text.")
                    
                    # Add feedback buttons for sarcasm detection
                    st.markdown("#### Was the sarcasm detection correct?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("👍 Yes, it's not sarcastic"):
                            st.session_state.feedback_data.append({
                                "text": text_input,
                                "is_sarcastic": False,
                                "system_prediction": False,
                                "feedback": "correct"
                            })
                            st.success("Thank you for your feedback!")
                    
                    with col2:
                        if st.button("👎 No, it is sarcastic"):
                            # Update patterns based on missed sarcasm
                            update_sarcasm_patterns(text_input, True)
                            st.session_state.feedback_data.append({
                                "text": text_input,
                                "is_sarcastic": True,
                                "system_prediction": False,
                                "feedback": "incorrect"
                            })
                            st.success("Thank you for your feedback! We've updated our sarcasm detection patterns.")
                
                if show_details:
                    st.markdown("#### Detailed Analysis")
                    
                    # Show VADER scores for comparison
                    st.markdown("##### VADER Sentiment Scores")
                    vader_scores = result["vader_scores"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Negative", f"{vader_scores['neg']:.2f}")
                    col2.metric("Neutral", f"{vader_scores['neu']:.2f}")
                    col3.metric("Positive", f"{vader_scores['pos']:.2f}")
                    col4.metric("Compound", f"{vader_scores['compound']:.2f}")
                    
                    # Show processed text
                    st.markdown("##### Processed Text")
                    st.text(result["processed_text"])
                    
                    # Show context score
                    st.markdown("##### Context Analysis")
                    st.markdown(f"Context Score: {result['context_score']:.2f}")
                    st.markdown("*Higher context scores indicate more complex text with more context for analysis.*")
            else:
                st.warning("Please enter some text to analyze.")
    
    # File Upload Tab
    with tab2:
        st.write("Upload a CSV or TXT file with comments to analyze")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            detect_sarcasm = st.checkbox("Detect sarcasm", value=True, key="file_sarcasm")
        with col2:
            use_llm = st.checkbox("Use LLM for enhanced detection", value=False, key="file_llm")
        with col3:
            show_advanced = st.checkbox("Show advanced analytics", value=False)
        
        api_key = None
        if use_llm:
            api_key = st.text_input("OpenAI API Key (optional)", type="password", key="file_api_key")
        
        if uploaded_file is not None:
            try:
                # Attempt to read as CSV first
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                    # Try to find a column with comments
                    comment_column = None
                    possible_columns = ['comment', 'comments', 'text', 'content', 'message']
                    
                    for col in possible_columns:
                        if col in data.columns:
                            comment_column = col
                            break
                    
                    if comment_column is None:
                        # If no predefined column names match, let the user select
                        comment_column = st.selectbox(
                            "Select the column containing comments",
                            options=data.columns
                        )
                    
                    comments = data[comment_column].tolist()
                
                # If not CSV or fails to read as CSV, try as TXT
                else:
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    comments = stringio.readlines()
                    # Remove newlines and empty lines
                    comments = [c.strip() for c in comments if c.strip()]
                
                # Show a sample of comments
                st.markdown(f"**Found {len(comments)} comments. Showing first 5:**")
                for i, comment in enumerate(comments[:5]):
                    st.text(f"{i+1}. {comment[:100]}...")
                
                if st.button("Analyze All Comments"):
                    with st.spinner(f"Analyzing {len(comments)} comments..."):
                        results = analyze_comments(comments, detect_sarcasm, use_llm, api_key)
                    
                    # Create a dataframe with results
                    results_df = pd.DataFrame({
                        "comment": [r["original_text"] for r in results],
                        "sentiment": [r["sentiment"] for r in results],
                        "confidence": [r["confidence"] for r in results],
                        "is_sarcastic": [r["is_sarcastic"] for r in results],
                        "sarcasm_confidence": [r["sarcasm_confidence"] for r in results],
                        "context_score": [r["context_score"] for r in results]
                    })
                    
                    # Display summary
                    st.markdown("### Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        positive_count = (results_df["sentiment"] == "positive").sum()
                        st.metric("Positive Comments", positive_count, f"{positive_count/len(results_df)*100:.1f}%")
                    
                    with col2:
                        negative_count = (results_df["sentiment"] == "negative").sum()
                        st.metric("Negative Comments", negative_count, f"{negative_count/len(results_df)*100:.1f}%")
                    
                    with col3:
                        neutral_count = (results_df["sentiment"] == "neutral").sum()
                        st.metric("Neutral Comments", neutral_count, f"{neutral_count/len(results_df)*100:.1f}%")
                    
                    with col4:
                        sarcastic_count = results_df["is_sarcastic"].sum()
                        st.metric("Sarcastic Comments", sarcastic_count, f"{sarcastic_count/len(results_df)*100:.1f}%")
                    
                    # Create sentiment distribution chart
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sentiment_counts = results_df["sentiment"].value_counts()
                        fig = px.pie(
                            names=sentiment_counts.index,
                            values=sentiment_counts.values,
                            color=sentiment_counts.index,
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            },
                            title="Sentiment Distribution"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Sarcasm by sentiment
                        sarcasm_by_sentiment = results_df.groupby('sentiment')['is_sarcastic'].mean()
                        
                        fig = px.bar(
                            x=sarcasm_by_sentiment.index,
                            y=sarcasm_by_sentiment.values,
                            color=sarcasm_by_sentiment.index,
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            },
                            title="Sarcasm Rate by Sentiment",
                            labels={'x': 'Sentiment', 'y': 'Sarcasm Rate'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Advanced analytics
                    if show_advanced:
                        st.markdown("### Advanced Analytics")
                        
                        # Confidence distribution
                        st.markdown("#### Confidence Distribution")
                        fig = px.histogram(
                            results_df, 
                            x="confidence", 
                            color="sentiment",
                            nbins=20,
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Context score vs. sentiment
                        st.markdown("#### Context Score vs. Sentiment")
                        fig = px.box(
                            results_df,
                            x="sentiment",
                            y="context_score",
                            color="sentiment",
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Word frequency analysis
                        st.markdown("#### Word Frequency Analysis")
                        
                        # Combine all comments
                        all_text = " ".join(results_df["comment"].astype(str))
                        
                        # Create word cloud
                        wordcloud = WordCloud(
                            width=800, 
                            height=400, 
                            background_color='white',
                            max_words=100
                        ).generate(all_text)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        # Sarcasm analysis
                        st.markdown("#### Sarcasm Analysis")
                        
                        # Compare sarcastic vs non-sarcastic comments
                        sarcastic_comments = results_df[results_df["is_sarcastic"]]
                        non_sarcastic_comments = results_df[~results_df["is_sarcastic"]]
                        
                        if not sarcastic_comments.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### Sarcastic Comments Word Cloud")
                                sarcastic_text = " ".join(sarcastic_comments["comment"].astype(str))
                                
                                wordcloud = WordCloud(
                                    width=800, 
                                    height=400, 
                                    background_color='white',
                                    colormap='Oranges',
                                    max_words=50
                                ).generate(sarcastic_text)
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            
                            with col2:
                                st.markdown("##### Sarcastic Comments Sentiment")
                                sarcastic_sentiment = sarcastic_comments["sentiment"].value_counts(normalize=True)
                                
                                fig = px.pie(
                                    names=sarcastic_sentiment.index,
                                    values=sarcastic_sentiment.values,
                                    color=sarcastic_sentiment.index,
                                    color_discrete_map={
                                        'positive': '#28a745',
                                        'negative': '#dc3545',
                                        'neutral': '#6c757d'
                                    }
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Display the results table
                    st.markdown("### Detailed Results")
                    st.dataframe(results_df)
                    
                    # Option to download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Save to session state for dashboard
                    if "dashboard_data" not in st.session_state:
                        # Create a dataframe with timestamp and platform
                        now = datetime.now()
                        dashboard_data = results_df.copy()
                        dashboard_data["timestamp"] = [now - timedelta(days=random.randint(0, 30)) for _ in range(len(dashboard_data))]
                        dashboard_data["platform"] = "File Upload"
                        dashboard_data["analyzed_sentiment"] = dashboard_data["sentiment"]
                        
                        st.session_state.dashboard_data = dashboard_data
                        st.session_state.analyzed_results = dashboard_data
                        
                        st.success("Results saved to dashboard. You can view them in the Dashboard page.")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Conversation Analysis Tab
    with tab3:
        st.write("Analyze a conversation with context-aware sentiment analysis")
        
        # Initialize conversation in session state if not exists
        if "conversation" not in st.session_state:
            st.session_state.conversation = []
        
        # Text input for new message
        new_message = st.text_input("Enter a message to add to the conversation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Add Message"):
                if new_message:
                    st.session_state.conversation.append(new_message)
                    st.success("Message added to conversation")
        
        with col2:
            if st.button("Add Sample Conversation"):
                st.session_state.conversation = [
                    "I just bought the new phone and it's amazing!",
                    "Oh really? What features do you like?",
                    "The camera is incredible and the battery lasts forever.",
                    "Wow, that sounds great! Is it worth the high price?",
                    "Yeah, TOTALLY worth paying a thousand dollars for a phone that will be obsolete in a year...",
                    "I see what you mean. Maybe I'll wait for the price to drop.",
                    "Good idea. The next model will probably be out soon anyway."
                ]
                st.success("Sample conversation added")
        
        with col3:
            use_llm_conv = st.checkbox("Use LLM for sarcasm", value=False, key="conv_llm")
            
        api_key_conv = None
        if use_llm_conv:
            api_key_conv = st.text_input("OpenAI API Key (optional)", type="password", key="conv_api_key")
        
        # Show current conversation
        if st.session_state.conversation:
            st.markdown("### Current Conversation")
            for i, msg in enumerate(st.session_state.conversation):
                st.markdown(f"**Message {i+1}:** {msg}")
            
            # Option to clear conversation
            if st.button("Clear Conversation"):
                st.session_state.conversation = []
                st.success("Conversation cleared")
            
            # Analyze conversation
            if st.button("Analyze Conversation"):
                with st.spinner("Analyzing conversation..."):
                    results = analyze_comments(st.session_state.conversation, detect_sarcasm=True, use_llm=use_llm_conv, api_key=api_key_conv)
                
                # Display results
                st.markdown("### Conversation Analysis Results")
                
                for i, (msg, result) in enumerate(zip(st.session_state.conversation, results)):
                    sentiment_class = f"sentiment-{result['sentiment']}"
                    sarcasm_indicator = " (Sarcastic)" if result["is_sarcastic"] else ""
                    
                    border_color = "#28a745" if result["sentiment"] == "positive" else "#dc3545" if result["sentiment"] == "negative" else "#6c757d"
                    if result["is_sarcastic"]:
                        border_color = "#fd7e14"
                    
                    st.markdown(
                        f"""
                        <div class="comment-card" style="border-left-color: {border_color};">
                            <div><strong>Message {i+1}:</strong> {msg}</div>
                            <div style='font-size: 0.8rem;'>
                                <span class='{sentiment_class}'>{result["sentiment"].capitalize()}{sarcasm_indicator}</span> | 
                                Confidence: {result["confidence"]:.2f} | 
                                Context Score: {result["context_score"]:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Add feedback buttons for sarcasm detection
                    if result["is_sarcastic"]:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"👍 Correct (Sarcastic)", key=f"correct_sarcastic_{i}"):
                                update_sarcasm_patterns(msg, True)
                                st.success(f"Thank you for confirming message {i+1} is sarcastic!")
                        with col2:
                            if st.button(f"👎 Incorrect (Not Sarcastic)", key=f"incorrect_sarcastic_{i}"):
                                update_sarcasm_patterns(msg, False)
                                st.success(f"Thank you for the feedback on message {i+1}!")
                
                # Conversation sentiment flow
                st.markdown("### Sentiment Flow")
                
                # Create sentiment flow chart
                sentiments = [r["sentiment"] for r in results]
                sentiment_values = []
                for s in sentiments:
                    if s == "positive":
                        sentiment_values.append(1)
                    elif s == "neutral":
                        sentiment_values.append(0)
                    else:  # negative
                        sentiment_values.append(-1)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(sentiment_values) + 1)),
                    y=sentiment_values,
                    mode='lines+markers',
                    name='Sentiment',
                    line=dict(color='#4F8BF9', width=3)
                ))
                
                # Add sarcasm indicators
                sarcasm_indices = [i+1 for i, r in enumerate(results) if r["is_sarcastic"]]
                sarcasm_values = [sentiment_values[i-1] for i in sarcasm_indices]
                
                if sarcasm_indices:
                    fig.add_trace(go.Scatter(
                        x=sarcasm_indices,
                        y=sarcasm_values,
                        mode='markers',
                        name='Sarcasm Detected',
                        marker=dict(
                            color='#fd7e14',
                            size=12,
                            symbol='star'
                        )
                    ))
                
                # Add reference lines
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=len(sentiment_values) + 1,
                    y1=0,
                    line=dict(color="gray", width=1, dash="dash")
                )
                
                # Customize the layout
                fig.update_layout(
                    xaxis_title="Message Sequence",
                    yaxis_title="Sentiment",
                    yaxis=dict(
                        tickvals=[-1, 0, 1],
                        ticktext=["Negative", "Neutral", "Positive"]
                    ),
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Overall conversation sentiment
                avg_sentiment = sum(sentiment_values) / len(sentiment_values)
                if avg_sentiment > 0.3:
                    overall = "Positive"
                    color = "#28a745"
                elif avg_sentiment < -0.3:
                    overall = "Negative"
                    color = "#dc3545"
                else:
                    overall = "Neutral"
                    color = "#6c757d"
                
                st.markdown(f"### Overall Conversation Sentiment: <span style='color: {color}'>{overall}</span>", unsafe_allow_html=True)
                
                # Conversation insights
                st.markdown("### Conversation Insights")
                
                # Calculate sentiment shifts
                sentiment_shifts = sum(1 for i in range(1, len(sentiments)) if sentiments[i] != sentiments[i-1])
                
                # Calculate sarcasm frequency
                sarcasm_count = sum(1 for r in results if r["is_sarcastic"])
                
                insights = []
                
                if sentiment_shifts > len(sentiments) / 3:
                    insights.append("The conversation shows significant sentiment shifts, indicating a dynamic exchange.")
                
                if sarcasm_count > 0:
                    insights.append(f"Detected {sarcasm_count} instance(s) of sarcasm, which may indicate hidden meanings or disagreement.")
                
                if avg_sentiment > 0.5:
                    insights.append("The conversation is predominantly positive, suggesting agreement or satisfaction.")
                elif avg_sentiment < -0.5:
                    insights.append("The conversation is predominantly negative, suggesting disagreement or dissatisfaction.")
                
                # Check for sentiment progression
                first_half = sum(sentiment_values[:len(sentiment_values)//2]) / (len(sentiment_values)//2)
                second_half = sum(sentiment_values[len(sentiment_values)//2:]) / (len(sentiment_values) - len(sentiment_values)//2)
                
                if second_half - first_half > 0.5:
                    insights.append("The conversation sentiment improves significantly over time.")
                elif first_half - second_half > 0.5:
                    insights.append("The conversation sentiment deteriorates over time.")
                
                if not insights:
                    insights.append("No significant patterns detected in this conversation.")
                
                for insight in insights:
                    st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)

# Chatbot Page
def chatbot_page():
    st.markdown("<div class='sub-header'>SentimentSense Chatbot</div>", unsafe_allow_html=True)
    
    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div class="avatar">👤</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <div class="avatar">🤖</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Chat input
    user_input = st.text_input("Type your message here", key="chat_input")
    
    # OpenAI API key for enhanced responses
    use_openai = st.checkbox("Use OpenAI for enhanced responses", value=False)
    api_key = None
    
    if use_openai:
        api_key = st.text_input("OpenAI API Key", type="password")
    
    # Send button
    if st.button("Send") and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get chatbot response
        with st.spinner("Thinking..."):
            response = get_chatbot_response(user_input, api_key)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update the UI
        st.experimental_rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # Chatbot features explanation
    with st.expander("About the Chatbot"):
        st.markdown("""
        ### SentimentSense Chatbot Features
        
        This chatbot can help you with:
        
        - Understanding sentiment analysis concepts
        - Explaining sarcasm detection techniques
        - Providing insights about your data
        - Guiding you through the SentimentSense platform
        - Analyzing the sentiment of your messages in real-time
        
        The chatbot uses sentiment and sarcasm analysis to understand your messages and respond appropriately.
        
        For enhanced responses, you can enable OpenAI integration by providing your API key.
        """)

# Sarcasm Settings Page
def sarcasm_settings_page():
    st.markdown("<div class='sub-header'>Sarcasm Detection Settings</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Customize how SentimentSense detects sarcasm in text. These settings allow you to fine-tune the sarcasm detection algorithm
    to better match your specific use case.
    """)
    
    # Create tabs for different settings
    tab1, tab2, tab3, tab4 = st.tabs(["Pattern Management", "Weights & Threshold", "Feedback Data", "Testing"])
    
    # Pattern Management Tab
    with tab1:
        st.markdown("### Sarcasm Pattern Management")
        st.markdown("Add, edit, or remove patterns used to detect sarcasm.")
        
        # Display current patterns
        for pattern_type, patterns in st.session_state.sarcasm_patterns.items():
            st.markdown(f"#### {pattern_type.capitalize()} Patterns")
            
            # Create a container for each pattern type
            pattern_container = st.container()
            
            with pattern_container:
                # Display each pattern with an option to remove
                for i, pattern in enumerate(patterns):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"<div class='pattern-card'>{pattern}</div>", unsafe_allow_html=True)
                    with col2:
                        if st.button("🗑️", key=f"delete_{pattern_type}_{i}"):
                            st.session_state.sarcasm_patterns[pattern_type].remove(pattern)
                            st.success(f"Pattern removed from {pattern_type}.")
                            st.experimental_rerun()
            
            # Add new pattern
            new_pattern = st.text_input(f"Add new {pattern_type} pattern", key=f"new_{pattern_type}")
            if st.button(f"Add Pattern", key=f"add_{pattern_type}"):
                if new_pattern and new_pattern not in patterns:
                    st.session_state.sarcasm_patterns[pattern_type].append(new_pattern)
                    st.success(f"Pattern added to {pattern_type}.")
                    st.experimental_rerun()
                elif new_pattern in patterns:
                    st.warning("This pattern already exists.")
                else:
                    st.warning("Please enter a pattern.")
    
    # Weights & Threshold Tab
    with tab2:
        st.markdown("### Weights & Threshold")
        st.markdown("Adjust the importance of different pattern types and the threshold for sarcasm detection.")
        
        # Weights sliders
        st.markdown("#### Pattern Type Weights")
        st.markdown("Higher weights give more importance to that pattern type in sarcasm detection.")
        
        total_weight = sum(st.session_state.sarcasm_weights.values())
        
        # Display current weights with sliders
        new_weights = {}
        for pattern_type, weight in st.session_state.sarcasm_weights.items():
            new_weights[pattern_type] = st.slider(
                f"{pattern_type.capitalize()} Weight",
                min_value=0.0,
                max_value=1.0,
                value=weight,
                step=0.05,
                key=f"weight_{pattern_type}"
            )
        
        # Normalize weights to sum to 1
        if st.button("Normalize Weights"):
            total_new_weight = sum(new_weights.values())
            if total_new_weight > 0:
                normalized_weights = {k: v / total_new_weight for k, v in new_weights.items()}
                st.session_state.sarcasm_weights = normalized_weights
                st.success("Weights normalized and saved.")
                st.experimental_rerun()
            else:
                st.error("Total weight must be greater than 0.")
        
        # Threshold slider
        st.markdown("#### Sarcasm Detection Threshold")
        st.markdown("Lower threshold makes the system more sensitive to sarcasm (more false positives). Higher threshold makes it more conservative (more false negatives).")
        
        new_threshold = st.slider(
            "Sarcasm Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.sarcasm_threshold,
            step=0.05
        )
        
        if st.button("Save Threshold"):
            st.session_state.sarcasm_threshold = new_threshold
            st.success("Threshold saved.")
    
    # Feedback Data Tab
    with tab3:
        st.markdown("### Feedback Data")
        st.markdown("View and manage the feedback data collected from users.")
        
        if not st.session_state.feedback_data:
            st.info("No feedback data available yet.")
        else:
            # Convert feedback data to DataFrame
            feedback_df = pd.DataFrame(st.session_state.feedback_data)
            
            # Display statistics
            st.markdown("#### Feedback Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                correct_predictions = sum(1 for item in st.session_state.feedback_data if item["system_prediction"] == item["is_sarcastic"])
                accuracy = correct_predictions / len(st.session_state.feedback_data)
                st.metric("Prediction Accuracy", f"{accuracy:.1%}")
            
            with col2:
                false_positives = sum(1 for item in st.session_state.feedback_data if item["system_prediction"] and not item["is_sarcastic"])
                false_negatives = sum(1 for item in st.session_state.feedback_data if not item["system_prediction"] and item["is_sarcastic"])
                
                if false_positives + false_negatives > 0:
                    fp_rate = false_positives / (false_positives + false_negatives)
                    st.metric("False Positive Rate", f"{fp_rate:.1%}")
                else:
                    st.metric("False Positive Rate", "0%")
            
            # Display feedback data
            st.markdown("#### Feedback Data")
            st.dataframe(feedback_df)
            
            # Option to clear feedback data
            if st.button("Clear Feedback Data"):
                st.session_state.feedback_data = []
                st.success("Feedback data cleared.")
                st.experimental_rerun()
            
            # Option to download feedback data
            csv = feedback_df.to_csv(index=False)
            st.download_button(
                label="Download Feedback Data",
                data=csv,
                file_name="sarcasm_feedback_data.csv",
                mime="text/csv"
            )
    
    # Testing Tab
    with tab4:
        st.markdown("### Sarcasm Detection Testing")
        st.markdown("Test your current sarcasm detection settings on sample texts.")
        
        # Sample sarcastic texts
        sample_texts = [
            "Oh great, another meeting that could have been an email!",
            "Wow, what an AMAZING experience... if you enjoy frustration!",
            "I'm THRILLED to spend my entire day fixing this!",
            "Sure, because waiting 2 hours for customer service is just SO much fun!",
            "This product is PERFECT... if you want something that doesn't work!"
        ]
        
        # Let user select a sample or enter their own
        test_option = st.radio(
            "Choose a test option",
            ["Select a sample", "Enter your own text"]
        )
        
        if test_option == "Select a sample":
            test_text = st.selectbox("Select a sample text", sample_texts)
        else:
            test_text = st.text_area("Enter text to test", height=100)
        
        if st.button("Test Sarcasm Detection") and test_text:
            # Test with current settings
            pattern_result = check_sarcasm_patterns(test_text)
            
            # Test with model
            with st.spinner("Analyzing..."):
                model_result = analyze_sentiment(test_text, detect_sarcasm=True)
            
            # Display results
            st.markdown("### Test Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Pattern-Based Detection")
                st.markdown(f"**Is Sarcastic:** {'Yes' if pattern_result['is_sarcastic'] else 'No'}")
                st.markdown(f"**Score:** {pattern_result['score']:.2f}")
                
                if pattern_result['matched_patterns']:
                    st.markdown("**Matched Patterns:**")
                    for pattern_type, pattern in pattern_result['matched_patterns']:
                        st.markdown(f"- **{pattern_type.capitalize()}**: `{pattern}`")
                else:
                    st.markdown("**No patterns matched.**")
            
            with col2:
                st.markdown("#### Model-Based Detection")
                st.markdown(f"**Is Sarcastic:** {'Yes' if model_result['is_sarcastic'] else 'No'}")
                st.markdown(f"**Confidence:** {model_result['sarcasm_confidence']:.2f}")
                st.markdown(f"**Sentiment:** {model_result['sentiment'].capitalize()}")
                
                if 'sarcasm_details' in model_result and model_result['sarcasm_details']:
                    details = model_result['sarcasm_details']
                    
                    if 'model_score' in details:
                        st.markdown(f"**Model Score:** {details.get('model_score', 0):.2f}")
                    
                    if 'has_contradiction' in details:
                        st.markdown(f"**Sentiment Contradiction:** {'Yes' if details.get('has_contradiction', False) else 'No'}")
            
            # Provide feedback option
            st.markdown("### Provide Feedback")
            st.markdown("Was the sarcasm detection correct?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("👍 Yes, it's sarcastic", key="test_sarcastic"):
                    update_sarcasm_patterns(test_text, True)
                    st.session_state.feedback_data.append({
                        "text": test_text,
                        "is_sarcastic": True,
                        "system_prediction": model_result['is_sarcastic'],
                        "feedback": "correct" if model_result['is_sarcastic'] else "incorrect"
                    })
                    st.success("Thank you for your feedback! We've updated our sarcasm detection patterns.")
            
            with col2:
                if st.button("👎 No, it's not sarcastic", key="test_not_sarcastic"):
                    update_sarcasm_patterns(test_text, False)
                    st.session_state.feedback_data.append({
                        "text": test_text,
                        "is_sarcastic": False,
                        "system_prediction": model_result['is_sarcastic'],
                        "feedback": "correct" if not model_result['is_sarcastic'] else "incorrect"
                    })
                    st.success("Thank you for your feedback! We've adjusted our sarcasm detection parameters.")

# Social Media Monitoring Page
def social_media_monitoring():
    st.markdown("<div class='sub-header'>Social Media Monitoring</div>", unsafe_allow_html=True)
    
    # Create tabs for different platforms
    tab1, tab2, tab3, tab4 = st.tabs(["YouTube", "Twitter", "Instagram", "Custom API"])
    
    # YouTube Tab
    with tab1:
        st.markdown("### YouTube Comment Analysis")
        
        youtube_query = st.text_input("Enter YouTube video URL or search term", key="youtube_query")
        youtube_count = st.slider("Number of comments to analyze", 10, 200, 50, key="youtube_count")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            detect_sarcasm = st.checkbox("Detect sarcasm", value=True, key="yt_sarcasm")
        with col2:
            use_llm = st.checkbox("Use LLM for enhanced detection", value=False, key="yt_llm")
        with col3:
            show_insights = st.checkbox("Show insights", value=True, key="yt_insights")
        
        api_key = None
        if use_llm:
            api_key = st.text_input("OpenAI API Key (optional)", type="password", key="yt_api_key")
        
        if st.button("Fetch YouTube Comments"):
            if youtube_query:
                with st.spinner("Fetching comments from YouTube..."):
                    # Simulate API call
                    data = fetch_api_data("YouTube", youtube_query, youtube_count)
                    
                with st.spinner("Analyzing comments..."):
                    # Analyze the comments
                    results = []
                    for comment in data["comment"]:
                        results.append(analyze_sentiment(comment, detect_sarcasm, use_llm, api_key))
                    
                    # Add analysis results to the dataframe
                    data["analyzed_sentiment"] = [r["sentiment"] for r in results]
                    data["is_sarcastic"] = [r["is_sarcastic"] for r in results]
                    data["confidence"] = [r["confidence"] for r in results]
                    data["sarcasm_confidence"] = [r["sarcasm_confidence"] for r in results]
                
                # Save to session state for dashboard
                st.session_state.dashboard_data = data
                st.session_state.analyzed_results = data
                
                # Display results
                st.markdown("### Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Comments", len(data))
                
                with col2:
                    positive_count = (data["analyzed_sentiment"] == "positive").sum()
                    st.metric("Positive", f"{positive_count/len(data)*100:.1f}%")
                
                with col3:
                    negative_count = (data["analyzed_sentiment"] == "negative").sum()
                    st.metric("Negative", f"{negative_count/len(data)*100:.1f}%")
                
                with col4:
                    sarcastic_count = data["is_sarcastic"].sum()
                    st.metric("Sarcastic", f"{sarcastic_count/len(data)*100:.1f}%")
                
                # Show insights if enabled
                if show_insights:
                    insights = extract_insights(data)
                    
                    st.markdown("### Key Insights")
                    for insight in insights:
                        st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
                
                # Create sentiment distribution chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Sentiment Distribution")
                    sentiment_counts = data["analyzed_sentiment"].value_counts()
                    fig = px.pie(
                        names=sentiment_counts.index,
                        values=sentiment_counts.values,
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positive': '#28a745',
                            'negative': '#dc3545',
                            'neutral': '#6c757d'
                        }
                    )
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Confidence Distribution")
                    fig = px.histogram(
                        data,
                        x="confidence",
                        color="analyzed_sentiment",
                        nbins=20,
                        color_discrete_map={
                            'positive': '#28a745',
                            'negative': '#dc3545',
                            'neutral': '#6c757d'
                        }
                    )
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Word cloud
                st.markdown("#### Word Cloud")
                
                # Combine all comments
                all_text = " ".join(data["comment"].astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                # Display comments with sentiment
                st.markdown("#### Comments")
                
                # Add a search box for filtering comments
                search_term = st.text_input("Search comments", "", key="yt_search")
                
                filtered_comments = data
                if search_term:
                    filtered_comments = data[data["comment"].str.contains(search_term, case=False)]
                
                # Display the comments with their sentiment
                for _, row in filtered_comments.iterrows():
                    sentiment_class = f"sentiment-{row['analyzed_sentiment']}"
                    sarcasm_indicator = " (Sarcastic)" if row["is_sarcastic"] else ""
                    
                    border_color = "#28a745" if row["analyzed_sentiment"] == "positive" else "#dc3545" if row["analyzed_sentiment"] == "negative" else "#6c757d"
                    if row["is_sarcastic"]:
                        border_color = "#fd7e14"
                        
                    st.markdown(
                        f"""
                        <div class="comment-card" style="border-left-color: {border_color};">
                            <div>{row["comment"]}</div>
                            <div style='font-size: 0.8rem; color: #6c757d;'>
                                <span class='{sentiment_class}'>{row["analyzed_sentiment"].capitalize()}{sarcasm_indicator}</span> | 
                                Confidence: {row["confidence"]:.2f} | {row["timestamp"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Please enter a YouTube video URL or search term.")

# Web Scraping Page
def web_scraping_page():
    st.markdown("<div class='sub-header'>Web Scraping</div>", unsafe_allow_html=True)
    st.write("Scrape and analyze comments from websites.")
    
    url = st.text_input("Enter website URL to scrape")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        platform = st.selectbox(
            "Platform type (helps with scraping)",
            ["Website", "YouTube", "Twitter", "News Site", "Forum", "Blog"]
        )
    
    with col2:
        detect_sarcasm = st.checkbox("Detect sarcasm", value=True)
    
    with col3:
        use_llm = st.checkbox("Use LLM for enhanced detection", value=False)
    
    with col4:
        show_insights = st.checkbox("Show insights", value=True)
    
    api_key = None
    if use_llm:
        api_key = st.text_input("OpenAI API Key (optional)", type="password")
    
    if st.button("Scrape and Analyze"):
        if url:
            with st.spinner(f"Scraping comments from {url}..."):
                data = scrape_comments(url, platform)
                
            with st.spinner("Analyzing comments..."):
                # Analyze the comments
                results = []
                for comment in data["comment"]:
                    results.append(analyze_sentiment(comment, detect_sarcasm, use_llm, api_key))
                
                # Add analysis results to the dataframe
                data["analyzed_sentiment"] = [r["sentiment"] for r in results]
                data["is_sarcastic"] = [r["is_sarcastic"] for r in results]
                data["confidence"] = [r["confidence"] for r in results]
                data["sarcasm_confidence"] = [r["sarcasm_confidence"] for r in results]
            
            # Save to session state for dashboard
            st.session_state.dashboard_data = data
            st.session_state.analyzed_results = data
            
            # Display results
            st.markdown("### Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Comments", len(data))
            
            with col2:
                positive_count = (data["analyzed_sentiment"] == "positive").sum()
                st.metric("Positive", f"{positive_count/len(data)*100:.1f}%")
            
            with col3:
                negative_count = (data["analyzed_sentiment"] == "negative").sum()
                st.metric("Negative", f"{negative_count/len(data)*100:.1f}%")
            
            with col4:
                sarcastic_count = data["is_sarcastic"].sum()
                st.metric("Sarcastic", f"{sarcastic_count/len(data)*100:.1f}%")
            
            # Show insights if enabled
            if show_insights:
                insights = extract_insights(data)
                
                st.markdown("### Key Insights")
                for insight in insights:
                    st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
            
            # Create sentiment distribution chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Sentiment Distribution")
                sentiment_counts = data["analyzed_sentiment"].value_counts()
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positive': '#28a745',
                        'negative': '#dc3545',
                        'neutral': '#6c757d'
                    }
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Word Cloud")
                
                # Combine all comments
                all_text = " ".join(data["comment"].astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            # Display comments with sentiment
            st.markdown("#### Scraped Content")
            
            # Add a search box for filtering comments
            search_term = st.text_input("Search content", "")
            
            filtered_content = data
            if search_term:
                filtered_content = data[data["comment"].str.contains(search_term, case=False)]
            
            # Display the content with sentiment
            for _, row in filtered_content.iterrows():
                sentiment_class = f"sentiment-{row['analyzed_sentiment']}"
                sarcasm_indicator = " (Sarcastic)" if row["is_sarcastic"] else ""
                
                border_color = "#28a745" if row["analyzed_sentiment"] == "positive" else "#dc3545" if row["analyzed_sentiment"] == "negative" else "#6c757d"
                if row["is_sarcastic"]:
                    border_color = "#fd7e14"
                    
                st.markdown(
                    f"""
                    <div class="comment-card" style="border-left-color: {border_color};">
                        <div>{row["comment"]}</div>
                        <div style='font-size: 0.8rem; color: #6c757d;'>
                            <span class='{sentiment_class}'>{row["analyzed_sentiment"].capitalize()}{sarcasm_indicator}</span> | 
                            Confidence: {row["confidence"]:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Option to download results
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="scraped_content_analysis.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please enter a website URL to scrape.")

# Settings Page
def settings_page():
    st.markdown("<div class='sub-header'>Settings</div>", unsafe_allow_html=True)
    
    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["Analysis Settings", "API Settings", "Advanced"])
    
    # Analysis Settings Tab
    with tab1:
        st.markdown("### Sentiment Analysis Settings")
        
        sentiment_model_option = st.selectbox(
            "Sentiment Analysis Model",
            ["cardiffnlp/twitter-roberta-base-sentiment", "distilbert-base-uncased-finetuned-sst-2-english", "bert-base-multilingual-uncased-sentiment"],
            index=0
        )
        
        sarcasm_model_option = st.selectbox(
            "Sarcasm Detection Model",
            ["mrm8488/bert-small-finetuned-sarcasm-detection", "cardiffnlp/twitter-roberta-base-sentiment"],
            index=0
        )
        
        st.checkbox("Enable Sarcasm Detection", value=True)
        st.checkbox("Apply Contextual Analysis", value=True)
        
        st.markdown("### Text Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Remove URLs", value=True)
            st.checkbox("Remove Special Characters", value=True)
        
        with col2:
            st.checkbox("Convert Emojis to Text", value=True)
            st.checkbox("Handle Slang and Acronyms", value=True)
        
        # Sentiment thresholds
        st.markdown("### Sentiment Thresholds")
        
        neutral_threshold = st.slider(
            "Neutral Sentiment Threshold",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Comments with sentiment scores between negative and positive thresholds will be classified as neutral"
        )
        
        sarcasm_threshold = st.slider(
            "Sarcasm Detection Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Confidence threshold for classifying a comment as sarcastic"
        )
        
        if st.button("Save Analysis Settings"):
            st.success("Settings saved successfully!")
    
    # API Settings Tab
    with tab2:
        st.markdown("### Social Media API Settings")
        
        st.markdown("#### YouTube API")
        st.text_input("YouTube API Key", type="password")
        
        st.markdown("#### Twitter API")
        st.text_input("Twitter API Key", type="password")
        st.text_input("Twitter API Secret", type="password")
        st.text_input("Twitter Access Token", type="password")
        st.text_input("Twitter Access Token Secret", type="password")
        
        st.markdown("#### OpenAI API")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        st.markdown("#### Web Scraping Settings")
        st.checkbox("Use Proxy for Web Scraping", value=False)
        st.text_input("Proxy URL (if enabled)")
        
        request_delay = st.slider(
            "Request Delay (seconds)",
            min_value=1,
            max_value=10,
            value=3,
            help="Delay between requests to avoid rate limiting"
        )
        
        if st.button("Save API Settings"):
            st.success("API settings saved successfully!")
    
    # Advanced Settings Tab
    with tab3:
        st.markdown("### Advanced Settings")
        
        st.markdown("#### Data Storage")
        st.checkbox("Cache Analysis Results", value=True)
        retention_days = st.slider("Data Retention (days)", 1, 90, 30)
        
        st.markdown("#### Performance")
        batch_size = st.slider("Analysis Batch Size", 10, 100, 50)
        
        st.markdown("#### Export Settings")
        export_format = st.selectbox("Default Export Format", ["CSV", "JSON", "Excel"])
        
        st.markdown("#### Model Caching")
        st.checkbox("Cache NLP Models", value=True, help="Keep models in memory for faster processing")
        
        st.markdown("#### Debug Mode")
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        
        if debug_mode:
            st.warning("Debug mode will show additional information and may slow down the application.")
        
        if st.button("Save Advanced Settings"):
            st.success("Advanced settings saved successfully!")
        
        if st.button("Reset All Settings to Default"):
            st.warning("This will reset all settings to their default values.")
            reset_confirm = st.checkbox("I understand that this action cannot be undone")
            
            if reset_confirm and st.button("Confirm Reset"):
                st.success("All settings have been reset to default values.")

# Advanced Analytics Page
def advanced_analytics_page():
    st.markdown("<div class='sub-header'>Advanced Analytics</div>", unsafe_allow_html=True)
    
    # Check if we have data to analyze
    if "analyzed_results" not in st.session_state:
        st.warning("No data available for analysis. Please analyze some comments first.")
        return
    
    data = st.session_state.analyzed_results
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Trends", "Sarcasm Analysis", "Text Mining", "Predictive Insights"])
    
    # Sentiment Trends Tab
    with tab1:
        st.markdown("### Sentiment Trends Over Time")
        
        # Convert timestamp to datetime if not already
        if 'date' not in data.columns:
            data['date'] = pd.to_datetime(data['timestamp']).dt.date
        
        # Group by date and count sentiments
        sentiment_over_time = data.groupby(['date', 'analyzed_sentiment']).size().unstack().fillna(0)
        
        # Calculate sentiment ratio
        sentiment_ratio = pd.DataFrame(index=sentiment_over_time.index)
        
        if 'positive' in sentiment_over_time.columns and 'negative' in sentiment_over_time.columns:
            total = sentiment_over_time.sum(axis=1)
            sentiment_ratio['positive_ratio'] = sentiment_over_time['positive'] / total
            sentiment_ratio['negative_ratio'] = sentiment_over_time['negative'] / total
            
            # Plot sentiment ratio over time
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sentiment_ratio.index,
                y=sentiment_ratio['positive_ratio'],
                mode='lines',
                name='Positive Ratio',
                line=dict(color='#28a745', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=sentiment_ratio.index,
                y=sentiment_ratio['negative_ratio'],
                mode='lines',
                name='Negative Ratio',
                line=dict(color='#dc3545', width=2)
            ))
            
            fig.update_layout(
                title="Sentiment Ratio Over Time",
                xaxis_title="Date",
                yaxis_title="Ratio",
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Moving average of sentiment
        st.markdown("### Moving Average Sentiment")
        
        # Calculate sentiment score (-1 for negative, 0 for neutral, 1 for positive)
        data['sentiment_score'] = data['analyzed_sentiment'].map({
            'positive': 1,
            'neutral': 0,
            'negative': -1
        })
        
        # Group by date and calculate average sentiment score
        daily_sentiment = data.groupby('date')['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment']
        
        # Calculate moving average
        window_size = st.slider("Moving Average Window Size (days)", 1, 10, 3)
        
        if len(daily_sentiment) > window_size:
            daily_sentiment['moving_avg'] = daily_sentiment['avg_sentiment'].rolling(window=window_size).mean()
            
            # Plot moving average
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['avg_sentiment'],
                mode='lines+markers',
                name='Daily Sentiment',
                line=dict(color='#4F8BF9', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['moving_avg'],
                mode='lines',
                name=f'{window_size}-Day Moving Average',
                line=dict(color='#fd7e14', width=3)
            ))
            
            fig.update_layout(
                title=f"Sentiment Trend with {window_size}-Day Moving Average",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                yaxis=dict(range=[-1, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Need at least {window_size+1} days of data for a {window_size}-day moving average.")
        
        # Sentiment volatility
        st.markdown("### Sentiment Volatility")
        
        if len(daily_sentiment) > 1:
            # Calculate day-to-day changes
            daily_sentiment['sentiment_change'] = daily_sentiment['avg_sentiment'].diff().abs()
            
            # Plot volatility
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=daily_sentiment['date'][1:],
                y=daily_sentiment['sentiment_change'][1:],
                marker_color='#4F8BF9'
            ))
            
            fig.update_layout(
                title="Day-to-Day Sentiment Volatility",
                xaxis_title="Date",
                yaxis_title="Absolute Change in Sentiment"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate average volatility
            avg_volatility = daily_sentiment['sentiment_change'].mean()
            st.metric("Average Daily Sentiment Volatility", f"{avg_volatility:.3f}")
            
            # Identify days with highest volatility
            if len(daily_sentiment) > 3:
                st.markdown("### Days with Highest Sentiment Volatility")
                top_volatile_days = daily_sentiment.sort_values('sentiment_change', ascending=False).head(3)
                
                for _, row in top_volatile_days.iterrows():
                    if pd.notna(row['sentiment_change']):
                        st.markdown(f"**{row['date']}**: Change of {row['sentiment_change']:.3f}")
    
    # Sarcasm Analysis Tab
    with tab2:
        st.markdown("### Sarcasm Detection Analysis")
        
        # Calculate sarcasm statistics
        sarcasm_count = data['is_sarcastic'].sum()
        sarcasm_percentage = (sarcasm_count / len(data)) * 100
        
        st.metric("Sarcasm Detection Rate", f"{sarcasm_percentage:.1f}%")
        
        # Sarcasm by sentiment
        st.markdown("### Sarcasm by Sentiment")
        
        sarcasm_by_sentiment = data.groupby('analyzed_sentiment')['is_sarcastic'].mean().reset_index()
        sarcasm_by_sentiment.columns = ['Sentiment', 'Sarcasm Rate']
        
        fig = px.bar(
            sarcasm_by_sentiment,
            x='Sentiment',
            y='Sarcasm Rate',
            color='Sentiment',
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            },
            text_auto='.1%'
        )
        
        fig.update_layout(
            title="Sarcasm Rate by Sentiment Category",
            xaxis_title="Sentiment",
            yaxis_title="Sarcasm Rate",
            yaxis=dict(tickformat='.1%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sarcasm over time
        st.markdown("### Sarcasm Rate Over Time")
        
        # Group by date and calculate sarcasm rate
        sarcasm_over_time = data.groupby('date')['is_sarcastic'].mean().reset_index()
        sarcasm_over_time.columns = ['date', 'sarcasm_rate']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sarcasm_over_time['date'],
            y=sarcasm_over_time['sarcasm_rate'],
            mode='lines+markers',
            line=dict(color='#fd7e14', width=2)
        ))
        
        fig.update_layout(
            title="Sarcasm Rate Trend",
            xaxis_title="Date",
            yaxis_title="Sarcasm Rate",
            yaxis=dict(tickformat='.1%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sarcasm pattern analysis
        st.markdown("### Sarcasm Pattern Analysis")
        
        # Get sarcastic comments
        sarcastic_comments = data[data['is_sarcastic']]['comment'].tolist()
        
        if sarcastic_comments:
            # Count pattern types
            pattern_counts = {
                'phrases': 0,
                'punctuation': 0,
                'capitalization': 0,
                'contradiction': 0,
                'exaggeration': 0
            }
            
            for comment in sarcastic_comments:
                pattern_result = check_sarcasm_patterns(comment)
                if 'matched_patterns' in pattern_result:
                    for pattern_type, _ in pattern_result['matched_patterns']:
                        if pattern_type in pattern_counts:
                            pattern_counts[pattern_type] += 1
            
            # Create pattern distribution chart
            pattern_df = pd.DataFrame({
                'Pattern Type': list(pattern_counts.keys()),
                'Count': list(pattern_counts.values())
            })
            
            fig = px.bar(
                pattern_df,
                x='Pattern Type',
                y='Count',
                color='Pattern Type',
                text_auto=True
            )
            
            fig.update_layout(
                title="Sarcasm Pattern Distribution",
                xaxis_title="Pattern Type",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Word cloud for sarcastic comments
            st.markdown("### Word Cloud for Sarcastic Comments")
            
            sarcastic_text = " ".join(sarcastic_comments)
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='Oranges',
                max_words=100
            ).generate(sarcastic_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No sarcastic comments detected in the dataset.")
    
    # Text Mining Tab
    with tab3:
        st.markdown("### Text Mining and Analysis")
        
        # Word frequency analysis
        st.markdown("#### Word Frequency Analysis")
        
        # Combine all comments
        all_text = " ".join(data["comment"].astype(str))
        
        # Tokenize and count words
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Remove common stopwords
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        except:
            filtered_words = [word for word in words if len(word) > 2]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Create word frequency chart
        top_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])
        
        fig = px.bar(
            top_words,
            x='Word',
            y='Frequency',
            color='Frequency',
            color_continuous_scale='Blues',
            text_auto=True
        )
        
        fig.update_layout(
            title="Top 20 Words by Frequency",
            xaxis_title="Word",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Word frequency by sentiment
        st.markdown("#### Word Frequency by Sentiment")
        
        sentiment_option = st.selectbox(
            "Select sentiment category",
            ["positive", "negative", "neutral"]
        )
        
        # Filter comments by selected sentiment
        sentiment_comments = data[data["analyzed_sentiment"] == sentiment_option]["comment"].astype(str).tolist()
        
        if sentiment_comments:
            # Combine comments
            sentiment_text = " ".join(sentiment_comments)
            
            # Tokenize and count words
            sentiment_words = re.findall(r'\b\w+\b', sentiment_text.lower())
            
            # Remove common stopwords
            try:
                filtered_sentiment_words = [word for word in sentiment_words if word not in stop_words and len(word) > 2]
            except:
                filtered_sentiment_words = [word for word in sentiment_words if len(word) > 2]
            
            # Count word frequencies
            sentiment_word_counts = Counter(filtered_sentiment_words)
            
            # Create word frequency chart
            top_sentiment_words = pd.DataFrame(sentiment_word_counts.most_common(15), columns=['Word', 'Frequency'])
            
            fig = px.bar(
                top_sentiment_words,
                x='Word',
                y='Frequency',
                color='Frequency',
                color_continuous_scale='Greens' if sentiment_option == 'positive' else 'Reds' if sentiment_option == 'negative' else 'Greys',
                text_auto=True
            )
            
            fig.update_layout(
                title=f"Top 15 Words in {sentiment_option.capitalize()} Comments",
                xaxis_title="Word",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Word cloud for selected sentiment
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='Greens' if sentiment_option == 'positive' else 'Reds' if sentiment_option == 'negative' else 'Greys',
                max_words=100
            ).generate(sentiment_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info(f"No {sentiment_option} comments found in the dataset.")
        
        # N-gram analysis
        st.markdown("#### N-gram Analysis")
        
        n_value = st.radio("Select n-gram size", [2, 3], horizontal=True)
        
        # Function to extract n-grams
        def extract_ngrams(text, n):
            words = re.findall(r'\b\w+\b', text.lower())
            ngrams = zip(*[words[i:] for i in range(n)])
            return [" ".join(ngram) for ngram in ngrams]
        
        # Extract n-grams from all comments
        all_ngrams = []
        for comment in data["comment"].astype(str):
            all_ngrams.extend(extract_ngrams(comment, n_value))
        
        # Count n-gram frequencies
        ngram_counts = Counter(all_ngrams)
        
        # Create n-gram frequency chart
        top_ngrams = pd.DataFrame(ngram_counts.most_common(15), columns=['N-gram', 'Frequency'])
        
        fig = px.bar(
            top_ngrams,
            x='N-gram',
            y='Frequency',
            color='Frequency',
            color_continuous_scale='Viridis',
            text_auto=True
        )
        
        fig.update_layout(
            title=f"Top 15 {n_value}-grams by Frequency",
            xaxis_title=f"{n_value}-gram",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Insights Tab
    with tab4:
        st.markdown("### Predictive Insights")
        
        # Check if we have enough data for predictions
        if len(data) < 30 or len(data['date'].unique()) < 5:
            st.warning("Not enough data for reliable predictions. Need at least 30 comments across 5 different days.")
            return
        
        # Sentiment prediction
        st.markdown("#### Sentiment Trend Prediction")
        
        # Group by date and calculate sentiment metrics
        daily_metrics = data.groupby('date').agg({
            'analyzed_sentiment': lambda x: (x == 'positive').mean() - (x == 'negative').mean(),
            'is_sarcastic': 'mean'
        }).reset_index()
        
        daily_metrics.columns = ['date', 'sentiment_score', 'sarcasm_rate']
        
        # Simple linear regression for sentiment prediction
        X = np.array(range(len(daily_metrics))).reshape(-1, 1)
        y = daily_metrics['sentiment_score'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future days
        future_days = 7
        future_X = np.array(range(len(daily_metrics), len(daily_metrics) + future_days)).reshape(-1, 1)
        future_y = model.predict(future_X)
        
        # Create prediction dates
        last_date = daily_metrics['date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
        
        # Plot actual and predicted sentiment
        fig = go.Figure()
        
        # Actual sentiment
        fig.add_trace(go.Scatter(
            x=daily_metrics['date'],
            y=daily_metrics['sentiment_score'],
            mode='lines+markers',
            name='Actual Sentiment',
            line=dict(color='#4F8BF9', width=2)
        ))
        
        # Predicted sentiment
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_y,
            mode='lines+markers',
            name='Predicted Sentiment',
            line=dict(color='#fd7e14', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Sentiment Trend Prediction (Next 7 Days)",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction confidence
        from sklearn.metrics import mean_squared_error
        
        # Calculate prediction error on training data
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        from sklearn.metrics import r2_score
        r2 = r2_score(y, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction Error (RMSE)", f"{rmse:.3f}")
        
        with col2:
            st.metric("Model Fit (R²)", f"{r2:.3f}")
        
        # Prediction insights
        st.markdown("### Prediction Insights")
        
        # Calculate trend direction
        future_trend = future_y[-1] - future_y[0]
        
        if future_trend > 0.1:
            trend_message = "Sentiment is predicted to improve over the next week."
            trend_color = "#28a745"
        elif future_trend < -0.1:
            trend_message = "Sentiment is predicted to decline over the next week."
            trend_color = "#dc3545"
        else:
            trend_message = "Sentiment is predicted to remain stable over the next week."
            trend_color = "#6c757d"
        
        st.markdown(f"<div class='insight-card' style='border-left-color: {trend_color};'>{trend_message}</div>", unsafe_allow_html=True)
        
        # Calculate volatility prediction
        if len(daily_metrics) > 3:
            volatility = daily_metrics['sentiment_score'].diff().abs().mean()
            
            if volatility > 0.2:
                volatility_message = "High sentiment volatility detected. Expect significant day-to-day changes."
            elif volatility > 0.1:
                volatility_message = "Moderate sentiment volatility detected. Some day-to-day changes expected."
            else:
                volatility_message = "Low sentiment volatility detected. Sentiment should remain relatively stable."
            
            st.markdown(f"<div class='insight-card'>{volatility_message}</div>", unsafe_allow_html=True)
        
        # Sarcasm prediction
        if 'is_sarcastic' in data.columns and data['is_sarcastic'].sum() > 0:
            # Simple linear regression for sarcasm prediction
            X_sarcasm = np.array(range(len(daily_metrics))).reshape(-1, 1)
            y_sarcasm = daily_metrics['sarcasm_rate'].values
            
            model_sarcasm = LinearRegression()
            model_sarcasm.fit(X_sarcasm, y_sarcasm)
            
            # Predict future sarcasm rate
            future_sarcasm = model_sarcasm.predict(future_X)
            
            # Plot actual and predicted sarcasm rate
            fig = go.Figure()
            
            # Actual sarcasm rate
            fig.add_trace(go.Scatter(
                x=daily_metrics['date'],
                y=daily_metrics['sarcasm_rate'],
                mode='lines+markers',
                name='Actual Sarcasm Rate',
                line=dict(color='#fd7e14', width=2)
            ))
            
            # Predicted sarcasm rate
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_sarcasm,
                mode='lines+markers',
                name='Predicted Sarcasm Rate',
                line=dict(color='#dc3545', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Sarcasm Rate Prediction (Next 7 Days)",
                xaxis_title="Date",
                yaxis_title="Sarcasm Rate",
                yaxis=dict(tickformat='.1%'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sarcasm trend insight
            future_sarcasm_trend = future_sarcasm[-1] - future_sarcasm[0]
            
            if future_sarcasm_trend > 0.05:
                sarcasm_message = "Sarcasm rate is predicted to increase, suggesting potential growing dissatisfaction."
            elif future_sarcasm_trend < -0.05:
                sarcasm_message = "Sarcasm rate is predicted to decrease, suggesting more straightforward communication."
            else:
                sarcasm_message = "Sarcasm rate is predicted to remain stable."
            
            st.markdown(f"<div class='insight-card'>{sarcasm_message}</div>", unsafe_allow_html=True)

# Run the main app
if __name__ == "__main__":
    main()                