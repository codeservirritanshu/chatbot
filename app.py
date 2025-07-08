import pandas as pd
import re
import random
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from waitress import serve
from tqdm import tqdm

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([word for word in text.split() if word not in stop_words])

# Load and clean dataset
df = pd.read_excel('Chatbot_dataset.xlsx')
df = df[~df['Intent'].str.startswith('Generated_Intent')]
df['Intent'] = df['Intent'].replace({
    r'^Greeting\d*$': 'Greeting',
    r'^Welcome\d*$': 'Welcome'
}, regex=True)
df['CleanedQuery'] = df['Query'].apply(clean_text)

# Use only one response per intent
responses = df.groupby('Intent')['Response'].first().to_dict()

# Load Sentence-BERT (PyTorch only)
print("🔍 Loading Sentence-BERT (MiniLM)...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, contextual
query_list = df['CleanedQuery'].tolist()
intent_list = df['Intent'].tolist()
query_embeddings = model.encode(query_list, convert_to_tensor=True)

# -----------------------------
# ✅ Evaluation (Top-1 Accuracy)
# -----------------------------
print("\n🔍 Evaluating model with Top-1 accuracy...")
correct = 0
total = len(query_list)

for i in tqdm(range(total), desc="Evaluating"):
    embedding = model.encode(query_list[i], convert_to_tensor=True)
    scores = util.cos_sim(embedding, query_embeddings)[0]
    best_idx = scores.argmax().item()
    if intent_list[best_idx] == intent_list[i]:
        correct += 1

accuracy = correct / total
print(f"\n✅ Semantic Top-1 Accuracy: {accuracy:.2%}\n")

# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    user_query = request.json.get('query', '')
    if not user_query.strip():
        return jsonify({'error': 'Query is empty'}), 400

    cleaned = clean_text(user_query)
    if not cleaned:
        return jsonify({'intent': 'unknown', 'response': "Please provide a meaningful question."})

    user_embedding = model.encode(cleaned, convert_to_tensor=True)
    scores = util.cos_sim(user_embedding, query_embeddings)[0]
    best_idx = scores.argmax().item()
    intent = intent_list[best_idx]
    response = responses.get(intent, "I'm still learning. Could you rephrase?")
    return jsonify({'intent': intent, 'response': response})

# -----------------------------
# Terminal Chat Mode
# -----------------------------
def chat_in_terminal():
    print("\n🤖 Chatbot (BERT-powered, no TensorFlow) — type 'exit' or 'sample'\n")
    while True:
        user_input = input("🗨️  You: ").strip()
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'sample':
            print("\n📋 Sample Q&A:")
            for _, row in df.sample(5).iterrows():
                print(f"\n🔹 Q: {row['Query']}\n   → 🤖 A: {row['Response']}")
            continue

        cleaned = clean_text(user_input)
        if not cleaned:
            print("⚠️  Please ask a valid question.")
            continue

        user_embedding = model.encode(cleaned, convert_to_tensor=True)
        scores = util.cos_sim(user_embedding, query_embeddings)[0]
        best_idx = scores.argmax().item()
        intent = intent_list[best_idx]
        response = responses.get(intent, "I'm still learning. Could you rephrase?")
        print(f"🤖 {response}")

# -----------------------------
# Launcher
# -----------------------------
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080) 