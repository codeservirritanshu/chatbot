import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk
from flask_cors import CORS
from flask import Flask, request, jsonify
from waitress import serve  

app = Flask(__name__)
CORS(app)  

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    df = pd.read_excel(r"generated_chatbot_data.xlsx") 
except Exception as e:
    print(f"Error loading the dataset: {e}")
    raise

df.drop(columns=['Chips'], axis=1, errors='ignore', inplace=True)

intents_summary = df.groupby('Intent').size()
print("Number of examples per intent:")
print(intents_summary)

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text) 
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['Query'] = df['Query'].apply(preprocess_text)

X = df['Query']
y = df['Intent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)

rf_y_pred = rf_model.predict(X_test_vec)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_y_pred))

lr_model = LogisticRegression(max_iter=200) 
lr_model.fit(X_train_vec, y_train)

lr_y_pred = lr_model.predict(X_test_vec)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_y_pred))

@app.route('/get_response', methods=['POST'])
def api_get_response():
    user_query = request.json['query']
    
    user_query_preprocessed = preprocess_text(user_query)
    user_query_tfidf = vectorizer.transform([user_query_preprocessed])
    
    predicted_intent = lr_model.predict(user_query_tfidf)[0]
    
    response_row = df[df['Intent'] == predicted_intent]
    if not response_row.empty:
        response = response_row['Response'].iloc[0]
    else:
        response = "I'm sorry, I didn't understand that. Could you rephrase?"
    
    return jsonify({'response': response})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080) 
