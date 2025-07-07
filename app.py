import pandas as pd
import re
import nltk
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Setup Flask app
app = Flask(__name__)
CORS(app)

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load dataset
try:
    df = pd.read_excel("generated_chatbot_data.xlsx")
except Exception as e:
    print(f"Error loading the dataset: {e}")
    raise

df.drop(columns=['Chips'], axis=1, errors='ignore', inplace=True)

# Merge similar or low-sample intents
merge_map = {
    'Greeting1': 'Greeting',
    'Greeting3': 'Greeting',
    'Welcome2': 'Welcome',
    'Welcome_1': 'Welcome',
    'Leave_request_1': 'Leave_request',
    'Company History': 'Company_Info',
    'Company Name': 'Company_Info',
    'Services Offered': 'Company_Info',
    'Privacy Policy': 'Company_Info'
}
df['Intent'] = df['Intent'].replace(merge_map)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['Query'] = df['Query'].apply(preprocess_text)

X = df['Query']
y = df['Intent']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.95)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# SVD Dimensionality Reduction
svd = TruncatedSVD(n_components=100, random_state=42)
X_train_svd = svd.fit_transform(X_train_vec)
X_test_svd = svd.transform(X_test_vec)

# Encode target
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# XGBoost model
xgb_model = XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.2,
    max_depth=3,
    n_estimators=200,
    subsample=0.8,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_svd, y_train_enc)

# Evaluate
xgb_preds_enc = xgb_model.predict(X_test_svd)
xgb_preds = label_encoder.inverse_transform(xgb_preds_enc)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))

# Use the model for deployment
deployed_model = xgb_model

# Helper to detect low-content input
def is_meaningless(tfidf_vector):
    return tfidf_vector.nnz == 0

# Unified response generator
def generate_bot_response(user_query):
    user_query_preprocessed = preprocess_text(user_query)
    user_query_tfidf = vectorizer.transform([user_query_preprocessed])

    if is_meaningless(user_query_tfidf):
        return "I'm sorry, I couldn't understand that. Could you rephrase?"

    user_query_svd = svd.transform(user_query_tfidf)

    probs = deployed_model.predict_proba(user_query_svd)[0]
    confidence = np.max(probs)
    predicted_label_enc = np.argmax(probs)
    predicted_intent = label_encoder.inverse_transform([predicted_label_enc])[0]

    if confidence < 0.8:
        return "I'm not sure I understand. Can you rephrase your question?"
    else:
        response_row = df[df['Intent'] == predicted_intent]
        if not response_row.empty:
            return response_row['Response'].iloc[0]
        else:
            return "Sorry, I couldn't find a proper response for that."

@app.route('/get_response', methods=['POST'])
def api_get_response():
    user_query = request.json.get('query', '')
    response = generate_bot_response(user_query)
    return jsonify({'response': response})

def chat_in_terminal():
    print("Chatbot is ready! Type your query (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = generate_bot_response(user_input)
        print("Bot:", response)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
