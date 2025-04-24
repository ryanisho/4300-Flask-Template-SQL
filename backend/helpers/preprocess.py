import pandas as pd
import numpy as np
import pickle
import nltk
import json
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

try:
    from nltk.tokenize import PunktSentenceTokenizer
    tokenizer = PunktSentenceTokenizer()
except:
    nltk.download('punkt', download_dir=nltk.data.path[0], quiet=False)
    print("Downloaded punkt to:", nltk.data.path[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_genre(genre):
    if isinstance(genre, list) and len(genre) > 0:
        if isinstance(genre[0], dict) and 'description' in genre[0]:
            return genre[0]['description']
        return str(genre[0])

    if isinstance(genre, str) and genre.startswith('[') and genre.endswith(']'):
        try:
            import ast
            parsed = ast.literal_eval(genre)
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict) and 'description' in parsed[0]:
                    return parsed[0]['description']
                return str(parsed[0])
        except:
            pass

    return genre


def load_and_clean_data():
    with open('data/books.json', 'r') as f:
        books_data = json.load(f)
    with open('data/movies.json', 'r') as f:
        movies_data = json.load(f)
    with open('data/games.json', 'r') as f:
        games_data = json.load(f)

    books_df = pd.DataFrame(books_data)
    movies_df = pd.DataFrame(movies_data)
    games_df = pd.DataFrame(games_data)

    books_df = books_df.drop_duplicates(subset=['book_name'])

    books_clean = books_df[['book_name', 'summaries', 'categories']].rename(
        columns={'book_name': 'title',
                 'summaries': 'description', 'categories': 'genre'}
    )
    books_clean['media_type'] = 'book'

    movies_clean = movies_df[['Series_Title', 'Overview', 'Genre']].rename(
        columns={'Series_Title': 'title',
                 'Overview': 'description', 'Genre': 'genre'}
    )
    movies_clean['media_type'] = 'movie'

    if 'genres' in games_df.columns:
        games_df['genres'] = games_df['genres'].apply(process_genre)

    games_df['description'] = games_df['short_description'].fillna(
        games_df['about_the_game']).fillna(
        games_df['detailed_description'])
    games_clean = games_df[['name', 'description', 'genres']].rename(
        columns={'name': 'title', 'genres': 'genre'}
    )
    games_clean['media_type'] = 'game'

    combined_df = pd.concat(
        [books_clean, movies_clean, games_clean], ignore_index=True)

    combined_df['description'] = combined_df['description'].fillna('')
    combined_df['genre'] = combined_df['genre'].fillna('')

    combined_df = combined_df[combined_df['description'].str.len() > 5]

    return combined_df


def preprocess_text(text):
    if not isinstance(text, str) or not text:
        return ""

    try:
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = word_tokenize(text.lower())
        filtered_tokens = [stemmer.stem(
            word) for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)
    except LookupError:
        print("Warning: Using fallback tokenization method")
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = text.lower().split()
        filtered_tokens = [stemmer.stem(
            word) for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)


def create_tfidf_svd_embeddings(df, n_components=300):
    print("Creating TF-IDF embeddings with SVD...")

    # Emphasize title more by repeating it
    df['combined_text'] = df['title'] + " " + df['title'] + " " + df['description']
    df['processed_text'] = [preprocess_text(
        text) for text in tqdm(df['combined_text'])]

    # Increase max_features for more vocabulary coverage
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased from 5000
        min_df=2,            # Reduced from 3 to capture more rare terms
        max_df=0.95,         # Increased from 0.9 to include more common terms
        sublinear_tf=True
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    explained_variance = svd.explained_variance_ratio_.sum()
    print(
        f"Explained variance with {n_components} components: {explained_variance:.4f}")

    embeddings = svd_matrix.tolist()
    df['tfidf_svd_embedding'] = embeddings

    with open('models/tfidf_svd_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'vectorizer': tfidf_vectorizer,
            'svd': svd,
            'feature_names': tfidf_vectorizer.get_feature_names_out(),
            'explained_variance': explained_variance
        }, f)

    return df, tfidf_vectorizer, svd


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_bert_embeddings(df, batch_size=16, max_length=512):
    print("Creating BERT embeddings with improved model...")

    # Use a better transformer model - mpnet has better semantic understanding
    model_name = "sentence-transformers/all-mpnet-base-v2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Process titles and descriptions separately to apply different weights
    df['title_text'] = df['title']
    df['desc_text'] = df['description']

    # Process titles
    print("Processing titles...")
    title_embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['title_text'].iloc[i:i+batch_size].tolist()

        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)

        batch_embs = mean_pooling(
            model_output, encoded_input['attention_mask']).cpu().numpy()
        title_embeddings.extend(batch_embs.tolist())

    # Process descriptions
    print("Processing descriptions...")
    desc_embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['desc_text'].iloc[i:i+batch_size].fillna('').tolist()

        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)

        batch_embs = mean_pooling(
            model_output, encoded_input['attention_mask']).cpu().numpy()
        desc_embeddings.extend(batch_embs.tolist())

    combined_embeddings = []
    for i in range(len(df)):
        title_emb = np.array(title_embeddings[i])
        desc_emb = np.array(desc_embeddings[i])
        
        title_emb = title_emb / (np.linalg.norm(title_emb) + 1e-8)
        desc_emb = desc_emb / (np.linalg.norm(desc_emb) + 1e-8)
        
        weighted_emb = (title_emb * 0.6) + (desc_emb * 0.4)
        
        weighted_emb = weighted_emb / (np.linalg.norm(weighted_emb) + 1e-8)
        
        combined_embeddings.append(weighted_emb.tolist())

    df['bert_embedding'] = combined_embeddings

    return df


def create_combined_embeddings(df, alpha=0.7):  # Increased BERT weight from 0.5 to 0.7
    tfidf_svd_dim = len(df['tfidf_svd_embedding'].iloc[0])
    bert_dim = len(df['bert_embedding'].iloc[0])
    print(f"Dimensions - TF-IDF+SVD: {tfidf_svd_dim}, BERT: {bert_dim}")

    combined_embeddings = []
    all_concatenated = []

    for i in range(len(df)):
        tfidf_svd_emb = np.array(df['tfidf_svd_embedding'].iloc[i])
        bert_emb = np.array(df['bert_embedding'].iloc[i])

        # Normalize
        tfidf_svd_emb = tfidf_svd_emb / (np.linalg.norm(tfidf_svd_emb) + 1e-8)
        bert_emb = bert_emb / (np.linalg.norm(bert_emb) + 1e-8)
        
        # Apply weights - more weight to BERT for better semantic matching
        tfidf_svd_emb = tfidf_svd_emb * (1 - alpha)  # 30% weight
        bert_emb = bert_emb * alpha  # 70% weight

        concatenated = np.concatenate([tfidf_svd_emb, bert_emb])
        all_concatenated.append(concatenated)

    all_concatenated = np.array(all_concatenated)

    # Increase dimensions for better information preservation
    final_dim = 768
    pca = PCA(n_components=final_dim)
    reduced_embeddings = pca.fit_transform(all_concatenated)

    print(
        f"Explained variance with PCA: {sum(pca.explained_variance_ratio_):.4f}")
    print(f"Final embedding dimension: {final_dim}")

    df['combined_embedding'] = reduced_embeddings.tolist()

    with open('models/embedding_pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    return df


def save_final_embeddings(df):
    save_df = df[['title', 'description', 'genre',
                  'media_type', 'combined_embedding']]

    with open('models/final_embeddings.pkl', 'wb') as f:
        pickle.dump(save_df, f)

    print(f"Saved embeddings for {len(df)} items")


if __name__ == "__main__":

    combined_df = load_and_clean_data()

    combined_df, vectorizer, svd = create_tfidf_svd_embeddings(
        combined_df, n_components=300)

    combined_df = create_bert_embeddings(combined_df)

    combined_df = create_combined_embeddings(combined_df, alpha=0.7)

    save_final_embeddings(combined_df)