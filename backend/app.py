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
        title_emb = mean_pooling(title_output, title_input['attention_mask']).cpu().numpy()[0]
        
        # Get description embedding (would be different in a real app)
        desc_output = model(**desc_input)
        desc_emb = mean_pooling(desc_output, desc_input['attention_mask']).cpu().numpy()[0]
    
    # Normalize embeddings
    query_tfidf_svd = query_tfidf_svd / (np.linalg.norm(query_tfidf_svd) + 1e-8)
    title_emb = title_emb / (np.linalg.norm(title_emb) + 1e-8)
    desc_emb = desc_emb / (np.linalg.norm(desc_emb) + 1e-8)
    
    # Weight the transformer embeddings (70% title, 30% description)
    weighted_transformer_emb = (title_emb * 0.7) + (desc_emb * 0.3)
    weighted_transformer_emb = weighted_transformer_emb / (np.linalg.norm(weighted_transformer_emb) + 1e-8)
    
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
        filtered_scores = [score if mask else 0 for score, mask in zip(cos_scores, media_mask)]
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
            boosted_score = 1.0 - ((1.0 - original_score) ** 1.5)  # Boost high scores
        else:
            boosted_score = original_score * 0.95  # Slightly lower poor scores
            
        # Get the adjustment factor for this rank
        adjustment_factor = base_adjustments[i] if i < len(base_adjustments) else 0.60
        
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
        final_score = (boosted_score * original_weight) + (rank_adjusted_score * rank_weight)
        
        # Add small random variation to prevent exact same scores (0-1% difference)
        if i > 0:  # Don't modify the top score
            variation = 0.01 * (0.5 - np.random.random())  # Random value between -0.005 and +0.005
            final_score = max(0, min(1, final_score + variation))  # Keep between 0-1
        
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
            'score': float(final_score)
        })

    return jsonify({'results': results})


@app.route('/explain', methods=['GET'])
def explain_recommendation():
    """
    Extract relevant SVD dimensions as tags with better filtering
    """
    item_id = request.args.get('id')
    query = request.args.get('query')

    if not item_id or not query:
        return jsonify({'error': 'Missing item_id or query parameter'})

    item_id = int(item_id)
    item = df.iloc[item_id]
    
    # Initialize stemmer
    stemmer = PorterStemmer()
    
    # Extract key terms from query and item for relevance checking
    query_lower = query.lower()
    stop_words = set(stopwords.words('english'))
    query_terms = set(word for word in query_lower.split() 
                     if len(word) > 3 and word not in stop_words)
    
    item_text = (item['title'] + " " + item['description']).lower()
    item_terms = set(word for word in item_text.split() 
                    if len(word) > 3 and word not in stop_words)
    
    # Process query to get its SVD representation
    processed_query = preprocess_text(query)
    query_tfidf = vectorizer.transform([processed_query])
    query_svd = svd.transform(query_tfidf)[0]
    
    # Get the SVD components (topics)
    svd_components = svd.components_
    feature_names = vectorizer.get_feature_names_out()
    
    # Get ALL SVD dimensions sorted by importance in query
    sorted_dims = np.argsort(-np.abs(query_svd))
    
    # Initialize list for filtered dimensions
    relevant_dimensions = []
    
    # Check more dimensions to find relevant ones
    for dim_idx in sorted_dims[:20]:  # Check top 20 dimensions
        # Skip if weight is too low
        dim_weight = query_svd[dim_idx]
        if abs(dim_weight) < 0.05:
            continue
            
        component = svd_components[dim_idx]
        
        # Get top terms for this dimension
        top_word_indices = np.argsort(-np.abs(component))[:10]
        top_terms = [feature_names[i] for i in top_word_indices if i < len(feature_names)]
        
        # Check if any top terms are relevant to query or item
        relevant_terms = []
        for term in top_terms:
            # Check if term is in query or item text directly
            if term in query_lower or term in item_text:
                relevant_terms.append(term)
                continue
                
            # Try stemming check
            try:
                stemmed_term = stemmer.stem(term)
                term_in_query = any(stemmed_term in stemmer.stem(qt) for qt in query_terms)
                term_in_item = any(stemmed_term in stemmer.stem(it) for it in item_terms)
                
                if term_in_query or term_in_item:
                    relevant_terms.append(term)
            except:
                # If stemming fails, just check direct inclusion
                if term in query_lower or term in item_text:
                    relevant_terms.append(term)
        
        # Only keep dimension if it has at least 1 relevant term
        if len(relevant_terms) >= 1:
            dim_terms = []
            for i in top_word_indices[:5]:
                if i < len(feature_names):
                    word = feature_names[i]
                    weight = component[i]
                    if abs(weight) > 0.01:
                        dim_terms.append({
                            "term": word,
                            "weight": float(abs(weight)),
                            "direction": "positive" if weight > 0 else "negative"
                        })
            
            # Create dimension object
            if dim_terms:
                dimension_name = ", ".join([t["term"] for t in dim_terms[:3]])
                relevant_dimensions.append({
                    "dimension": int(dim_idx),
                    "name": f"Dimension {dim_idx}: {dimension_name}",
                    "weight": float(abs(dim_weight)),
                    "terms": dim_terms
                })
        
        # Stop when we have enough relevant dimensions
        if len(relevant_dimensions) >= 3:
            break
    
    # If we don't have enough dimensions, just use the top dimensions by weight
    if len(relevant_dimensions) < 2:
        for dim_idx in sorted_dims[:5]:
            if dim_idx in [d["dimension"] for d in relevant_dimensions]:
                continue
                
            dim_weight = query_svd[dim_idx]
            component = svd_components[dim_idx]
            
            # Get top terms for this dimension
            top_word_indices = np.argsort(-np.abs(component))[:5]
            dim_terms = []
            for i in top_word_indices:
                if i < len(feature_names):
                    word = feature_names[i]
                    weight = component[i]
                    if abs(weight) > 0.01:
                        dim_terms.append({
                            "term": word,
                            "weight": float(abs(weight)),
                            "direction": "positive" if weight > 0 else "negative"
                        })
            
            if dim_terms:
                dimension_name = ", ".join([t["term"] for t in dim_terms[:3]])
                relevant_dimensions.append({
                    "dimension": int(dim_idx),
                    "name": f"Dimension {dim_idx}: {dimension_name}",
                    "weight": float(abs(dim_weight)),
                    "terms": dim_terms
                })
            
            if len(relevant_dimensions) >= 3:
                break
    
    # Sort by weight
    relevant_dimensions.sort(key=lambda x: x["weight"], reverse=True)
    
    # Calculate similarity score
    query_vector = generate_query_embedding(query)
    item_vector = embeddings[item_id]
    similarity = 1 - cosine(query_vector, item_vector)
    
    # Apply boosting for consistency
    if similarity > 0.5:
        boosted_similarity = 1.0 - ((1.0 - similarity) ** 1.5)
    else:
        boosted_similarity = similarity * 0.95
    
    explanation = {
        'title': item['title'],
        'media_type': item['media_type'],
        'svd_dimensions': relevant_dimensions[:3],
        'similarity_score': float(boosted_similarity)
    }

    return jsonify(explanation)


# Helper function to detect similar words
def are_similar_words(word1, word2):
    """Check if two words are similar enough to be considered duplicates"""
    # Exact match
    if word1 == word2:
        return True
    
    # One is substring of the other
    if word1 in word2 or word2 in word1:
        return True
    
    # Plural forms (simplified check)
    if word1 + 's' == word2 or word2 + 's' == word1:
        return True
    
    # Could add more similarity checks if needed
    
    return False


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5050)