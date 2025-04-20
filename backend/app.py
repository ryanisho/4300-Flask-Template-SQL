import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "admin"
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "kardashiandb"

mysql_engine = MySQLDatabaseHandler(
    LOCAL_MYSQL_USER, LOCAL_MYSQL_USER_PASSWORD, LOCAL_MYSQL_PORT, LOCAL_MYSQL_DATABASE)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


def preprocess_text(text: str) -> str:
    """
    Preprocess text with error handling for tokenization.

    Args:
        text (str): The input text to preprocess

    Returns:
        str: The preprocessed text with stopwords removed and words stemmed
    """
    if not isinstance(text, str) or not text:
        return ""

    try:
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = word_tokenize(text.lower())
        filtered_tokens = [
            stemmer.stem(word)
            for word in tokens
            if word.isalnum() and word not in stop_words
        ]
        return ' '.join(filtered_tokens)
    except LookupError:
        print("Warning: Using fallback tokenization method")
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = text.lower().split()
        filtered_tokens = [
            stemmer.stem(word)
            for word in tokens
            if word.isalnum() and word not in stop_words
        ]
        return ' '.join(filtered_tokens)


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to get sentence embeddings from BERT token embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


with open('./helpers/models/final_embeddings.pkl', 'rb') as f:
    df = pickle.load(f)

with open('./helpers/models/tfidf_svd_embeddings.pkl', 'rb') as f:
    tfidf_data = pickle.load(f)
    vectorizer = tfidf_data['vectorizer']
    svd = tfidf_data['svd']

with open('./helpers/models/embedding_pca.pkl', 'rb') as f:
    pca = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2').to(device)

embeddings = np.stack([np.array(emb)
                       for emb in df['combined_embedding'].values])


def generate_query_embedding(query):
    """
    Generate embedding for a search query using the new combined model
    """
    processed_query = preprocess_text(query)

    query_tfidf = vectorizer.transform([processed_query])
    query_tfidf_svd = svd.transform(query_tfidf)[0]

    encoded_input = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    query_bert_emb = mean_pooling(
        model_output, encoded_input['attention_mask']).cpu().numpy()[0]

    query_tfidf_svd = query_tfidf_svd / \
        (np.linalg.norm(query_tfidf_svd) + 1e-8)
    query_bert_emb = query_bert_emb / (np.linalg.norm(query_bert_emb) + 1e-8)

    concatenated = np.concatenate([query_tfidf_svd, query_bert_emb])
    query_vector = pca.transform([concatenated])[0]

    return query_vector


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    media_type = request.form.get('media_type', 'all')

    if not query:
        return jsonify({'error': 'No query provided'})

    query_vector = generate_query_embedding(query)

    cos_scores = [1 - cosine(query_vector, emb) for emb in embeddings]

    if media_type != 'all':
        media_mask = df['media_type'] == media_type
        filtered_scores = [score if mask else 0 for score,
                           mask in zip(cos_scores, media_mask)]
        cos_scores = filtered_scores

    top_results = np.argsort(-np.array(cos_scores))[:10]

    results = []
    for idx in top_results:
        results.append({
            'title': df.iloc[idx]['title'],
            'media_type': df.iloc[idx]['media_type'],
            'genre': df.iloc[idx]['genre'],
            'description': df.iloc[idx]['description'],
            'score': float(cos_scores[idx]),
            'rating': float(df.iloc[idx]['score']) if 'score' in df.columns else None
        })

    return jsonify({'results': results})


@app.route('/explain', methods=['GET'])
def explain_recommendation():
    item_id = request.args.get('id')
    query = request.args.get('query')

    if not item_id or not query:
        return jsonify({'error': 'Missing item_id or query parameter'})

    item_id = int(item_id)
    item = df.iloc[item_id]

    if item['media_type'] == 'game':
        if isinstance(item['genre'], list) and all(isinstance(genre, dict) for genre in item['genre']):
            genres = [genre.get('description', '') for genre in item['genre']]
            item['genre'] = ', '.join(genres)  # Join them into a single string

    processed_query = preprocess_text(query)

    query_tfidf = vectorizer.transform([processed_query]).toarray()[0]

    feature_names = vectorizer.get_feature_names_out()

    top_query_terms_idx = np.argsort(-query_tfidf)[:10]
    top_query_terms = [feature_names[idx]
                       for idx in top_query_terms_idx if query_tfidf[idx] > 0]

    query_vector = generate_query_embedding(query)
    item_vector = embeddings[item_id]
    similarity = 1 - cosine(query_vector, item_vector)

    explanation = {
        'title': item['title'],
        'media_type': item['media_type'],
        'matching_terms': top_query_terms,
        'similarity_score': float(similarity),
    }

    return jsonify(explanation)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5050)
