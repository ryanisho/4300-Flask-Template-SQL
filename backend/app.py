from sklearn.feature_extraction.text import TfidfVectorizer
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


def extract_top_keywords_from_top_results(top_indices, all_descriptions, top_n=10):
    """
    Compute contrastive TF-IDF scores between top result descriptions and the rest
    """
    top_descriptions = [all_descriptions[i] for i in top_indices]
    rest_indices = list(set(range(len(all_descriptions))) - set(top_indices))
    rest_descriptions = [all_descriptions[i] for i in rest_indices]

    top_doc = " ".join(top_descriptions)
    rest_doc = " ".join(rest_descriptions)

    corpus = [top_doc, rest_doc]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    top_scores = tfidf_matrix[0].toarray()[0]
    rest_scores = tfidf_matrix[1].toarray()[0]
    score_diff = top_scores - rest_scores

    feature_names = vectorizer.get_feature_names_out()
    top_indices = score_diff.argsort()[::-1][:top_n]

    keywords = [(feature_names[i], round(score_diff[i], 3))
                for i in top_indices]
    return keywords


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
    Mean pooling to get sentence embeddings from transformer token embeddings
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

# Use the better transformer model
model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

embeddings = np.stack([np.array(emb)
                       for emb in df['combined_embedding'].values])


def generate_query_embedding(query):
    """
    Generate embedding for a search query using the improved combined model
    with better transformer and weighted approach
    """
    # Process for TF-IDF + SVD
    processed_query = preprocess_text(query)
    query_tfidf = vectorizer.transform([processed_query])
    query_tfidf_svd = svd.transform(query_tfidf)[0]

    # Process for transformer - split into separate title-like and description-like components
    # Title-like: use the entire query as is
    # Description-like: use the entire query as is (in a real app, this might be more detailed)
    title_input = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    desc_input = title_input  # For simplicity, use the same input for both

    with torch.no_grad():
        # Get title embedding
        title_output = model(**title_input)
        title_emb = mean_pooling(
            title_output, title_input['attention_mask']).cpu().numpy()[0]

        # Get description embedding (would be different in a real app)
        desc_output = model(**desc_input)
        desc_emb = mean_pooling(
            desc_output, desc_input['attention_mask']).cpu().numpy()[0]

    # Normalize embeddings
    query_tfidf_svd = query_tfidf_svd / \
        (np.linalg.norm(query_tfidf_svd) + 1e-8)
    title_emb = title_emb / (np.linalg.norm(title_emb) + 1e-8)
    desc_emb = desc_emb / (np.linalg.norm(desc_emb) + 1e-8)

    # Weight the transformer embeddings (70% title, 30% description)
    weighted_transformer_emb = (title_emb * 0.7) + (desc_emb * 0.3)
    weighted_transformer_emb = weighted_transformer_emb / \
        (np.linalg.norm(weighted_transformer_emb) + 1e-8)

    # Weight between TFIDF+SVD and transformer (30% TFIDF, 70% transformer)
    alpha = 0.7
    query_tfidf_svd = query_tfidf_svd * (1 - alpha)
    weighted_transformer_emb = weighted_transformer_emb * alpha

    # Concatenate and transform with PCA
    concatenated = np.concatenate([query_tfidf_svd, weighted_transformer_emb])
    query_vector = pca.transform([concatenated])[0]

    return query_vector


@app.route('/')
def index():
    return render_template('base.html')


# Updated search function with more differentiated scores
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    media_type = request.form.get('media_type', 'all')

    if not query:
        return jsonify({'error': 'No query provided'})

    query_vector = generate_query_embedding(query)

    # Calculate base similarity scores
    cos_scores = [1 - cosine(query_vector, emb) for emb in embeddings]

    # Apply media type filtering if specified
    if media_type != 'all':
        media_mask = df['media_type'] == media_type
        filtered_scores = [score if mask else 0 for score,
                           mask in zip(cos_scores, media_mask)]
        cos_scores = filtered_scores

    # Get the top indices
    top_indices = np.argsort(-np.array(cos_scores))[:10]

    # Get the maximum score for reference
    max_score = cos_scores[top_indices[0]] if len(top_indices) > 0 else 0.0

    # Create a unique adjustment factor for each rank
    # Base adjustment factors (as percentage of top score)
    base_adjustments = [
        1.00,  # 1st item: 100% of top score
        0.92,  # 2nd item: 92% of top score
        0.87,  # 3rd item: 87% of top score
        0.83,  # 4th item: 83% of top score
        0.79,  # 5th item: 79% of top score
        0.76,  # 6th item: 76% of top score
        0.73,  # 7th item: 73% of top score
        0.70,  # 8th item: 70% of top score
        0.67,  # 9th item: 67% of top score
        0.64   # 10th item: 64% of top score
    ]

    results = []
    for i, idx in enumerate(top_indices):
        # Skip any zero scores (filtered out by media type)
        if cos_scores[idx] == 0:
            continue

        # Apply individual non-linear boosting first
        original_score = cos_scores[idx]
        if original_score > 0.5:
            boosted_score = 1.0 - ((1.0 - original_score)
                                   ** 1.5)  # Boost high scores
        else:
            boosted_score = original_score * 0.95  # Slightly lower poor scores

        # Get the adjustment factor for this rank
        adjustment_factor = base_adjustments[i] if i < len(
            base_adjustments) else 0.60

        # Calculate the rank-adjusted score
        rank_adjusted_score = max_score * adjustment_factor

        # Use original score influence based on its quality
        # Higher original scores should have more influence on final score
        if original_score > 0.7:  # Very good match
            # Strong original matches get more weight from their original score
            original_weight = 0.6
            rank_weight = 0.4
        elif original_score > 0.5:  # Good match
            # Good matches get balanced weighting
            original_weight = 0.5
            rank_weight = 0.5
        elif original_score > 0.3:  # Moderate match
            # Moderate matches lean more on rank adjustment
            original_weight = 0.3
            rank_weight = 0.7
        else:  # Poor match
            # Poor matches mostly use rank adjustment
            original_weight = 0.2
            rank_weight = 0.8

        # Blend boosted score with rank adjustment
        final_score = (boosted_score * original_weight) + \
            (rank_adjusted_score * rank_weight)

        # Add small random variation to prevent exact same scores (0-1% difference)
        if i > 0:  # Don't modify the top score
            # Random value between -0.005 and +0.005
            variation = 0.01 * (0.5 - np.random.random())
            # Keep between 0-1
            final_score = max(0, min(1, final_score + variation))

        # Safety check: avoid inflating very poor matches
        if original_score < 0.3 and final_score > 0.5:
            # If original score was very low, cap the boost
            final_score = min(final_score, 0.5)

        # Round to 2 decimal places for display
        final_score = round(final_score, 2)

        results.append({
            'title': df.iloc[idx]['title'],
            'media_type': df.iloc[idx]['media_type'],
            'genre': df.iloc[idx]['genre'],
            'description': df.iloc[idx]['description'],
            # Include the original media rating score
            'media_score': float(df.iloc[idx]['score']),
            'score': float(final_score)
        })

    return jsonify({'results': results})


@app.route('/explain', methods=['GET'])
def explain_recommendation():
    """
    Explain why a specific item was recommended by showing exact word matches
    between the query and the media content (title and description).
    """
    item_id = request.args.get('id')
    query = request.args.get('query')

    if not item_id or not query:
        return jsonify({'error': 'Missing item_id or query parameter'})

    try:
        item_id = int(item_id)
        item = df.iloc[item_id]
    except (ValueError, IndexError):
        return jsonify({'error': 'Invalid item ID'})

    # Initialize stemmer and stop words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Additional common words to filter out
    generic_terms = {'thing', 'make', 'look', 'just', 'like', 'know', 'time',
                     'good', 'really', 'great', 'way', 'find', 'part', 'take',
                     'much', 'even', 'first', 'new', 'one', 'two', 'many', 'also',
                     'get', 'use', 'may', 'well', 'come', 'give', 'every', 'day',
                     'year', 'back', 'today', 'lets', 'going', 'best'}

    # Process query into clean terms
    query_terms = []
    # Use word_tokenize for better tokenization
    for word in word_tokenize(query.lower()):
        if len(word) > 2 and word.isalpha() and word not in stop_words and word not in generic_terms:
            stemmed = stemmer.stem(word)
            query_terms.append({
                'original': word,
                'stemmed': stemmed
            })

    # Extract terms from the item title
    title_terms = []
    for word in word_tokenize(item['title'].lower()):
        if len(word) > 2 and word.isalpha() and word not in stop_words and word not in generic_terms:
            stemmed = stemmer.stem(word)
            title_terms.append({
                'original': word,
                'stemmed': stemmed,
                'source': 'title',
                'context': item['title']
            })

    # Extract terms from the item description
    desc_terms = []
    if isinstance(item['description'], str):
        for word in word_tokenize(item['description'].lower()):
            if len(word) > 2 and word.isalpha() and word not in stop_words and word not in generic_terms:
                stemmed = stemmer.stem(word)
                desc_terms.append({
                    'original': word,
                    'stemmed': stemmed,
                    'source': 'description',
                    'context': get_word_context(word, item['description'])
                })

    item_terms = title_terms + desc_terms

    # Find direct matching terms between query and item
    direct_matches = []
    matched_stems = set()  # Track matched stems to avoid duplicates

    for query_term in query_terms:
        for item_term in item_terms:
            # Check if stems match
            if query_term['stemmed'] == item_term['stemmed']:
                if item_term['stemmed'] in matched_stems:
                    continue

                direct_matches.append({
                    'query_term': query_term['original'],
                    'item_term': item_term['original'],
                    'source': item_term['source'],
                    'context': item_term['context'],
                    'match_type': 'exact' if query_term['original'] == item_term['original'] else 'stem',
                    'importance': 1.0 if item_term['source'] == 'title' else 0.8
                })

                matched_stems.add(item_term['stemmed'])

    # Find partial/substring matches
    partial_matches = []
    for query_term in query_terms:
        for item_term in item_terms:
            if item_term['stemmed'] in matched_stems:
                continue

            if is_substring_match(query_term['original'], item_term['original']):
                partial_matches.append({
                    'query_term': query_term['original'],
                    'item_term': item_term['original'],
                    'source': item_term['source'],
                    'context': item_term['context'],
                    'match_type': 'substring',
                    'importance': 0.7 if item_term['source'] == 'title' else 0.6
                })
                matched_stems.add(item_term['stemmed'])

    # Find semantic matches using our embedding model
    semantic_matches = []
    if len(direct_matches) + len(partial_matches) < 5:
        semantic_matches = find_semantic_matches(
            query, item, query_terms, item_terms, matched_stems)

    # Combine all matches and sort by importance
    all_matches = direct_matches + partial_matches + semantic_matches
    all_matches.sort(key=lambda x: x['importance'], reverse=True)

    # Limit to top 15 matches
    all_matches = all_matches[:15]

    # Calculate overall similarity score
    query_vector = generate_query_embedding(query)
    item_vector = embeddings[item_id]
    similarity = float(1 - cosine(query_vector, item_vector))
    similarity = round(similarity, 3)

    # Format response
    explanation = {
        'query': query,
    }

    # Extract top 10 related keywords from top results
    top_result_indices = np.argsort(
        -np.array([1 - cosine(query_vector, emb) for emb in embeddings]))[:10]
    all_descriptions = df['description'].fillna('').tolist()
    keyword_tags = extract_top_keywords_from_top_results(
        top_result_indices, all_descriptions, top_n=10)

    explanation['top_keywords'] = [
        {'keyword': k, 'score': s} for k, s in keyword_tags]

    print(explanation)

    return jsonify(explanation)


def get_word_context(word, text, window=20):
    """Extract a snippet of text surrounding the word for context"""
    if not text or not isinstance(text, str):
        return ""

    word_pos = text.lower().find(word.lower())
    if word_pos == -1:
        return ""

    start = max(0, word_pos - window)
    end = min(len(text), word_pos + len(word) + window)

    context = text[start:end]
    # Add ellipsis if we're not showing the full text
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."

    return context


def is_substring_match(word1, word2):
    """Check if two words have a substring relationship or are very similar"""
    # Skip if either is too short
    if len(word1) < 4 or len(word2) < 4:
        return False

    # One is substring of the other
    if word1 in word2 or word2 in word1:
        return True

    # Plural forms
    if word1 + 's' == word2 or word1 == word2 + 's':
        return True
    if word1 + 'es' == word2 or word1 == word2 + 'es':
        return True

    return False


def find_semantic_matches(query, item, query_terms, item_terms, matched_stems):
    """Find semantic matches between query and item using the embedding model"""
    semantic_matches = []

    # Get top important terms using SVD components
    processed_query = preprocess_text(query)
    query_tfidf = vectorizer.transform([processed_query])
    query_svd = svd.transform(query_tfidf)[0]

    # Get item embeddings
    item_title_processed = preprocess_text(item['title'])
    item_tfidf = vectorizer.transform([item_title_processed])
    item_svd = svd.transform(item_tfidf)[0]

    # Get top dimensions where query and item align
    query_top_dims = np.argsort(-np.abs(query_svd))[:5]
    item_top_dims = np.argsort(-np.abs(item_svd))[:5]

    # Find overlapping important dimensions
    overlapping_dims = set(query_top_dims).intersection(set(item_top_dims))

    # Use up to 3 top overlapping dimensions
    for dim_idx in list(overlapping_dims)[:3]:
        component = svd.components_[dim_idx]
        feature_names = vectorizer.get_feature_names_out()
        top_indices = np.argsort(-np.abs(component))[:10]

        for idx in top_indices:
            if idx >= len(feature_names):
                continue

            term = feature_names[idx]
            if len(term) <= 2:
                continue

            stemmed_term = stemmer.stem(term)

            if stemmed_term in matched_stems:
                continue

            # Check if term appears in item content
            for item_term in item_terms:
                if stemmed_term == item_term['stemmed']:
                    semantic_matches.append({
                        'query_term': 'semantic concept',
                        'item_term': item_term['original'],
                        'source': item_term['source'],
                        'context': item_term['context'],
                        'match_type': 'semantic',
                        'importance': 0.5 if item_term['source'] == 'title' else 0.4,
                        'concept': f"Concept #{dim_idx+1}"
                    })

                    matched_stems.add(stemmed_term)
                    break

    return semantic_matches


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5050)
