from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json, torch, secrets
from sklearn.manifold import TSNE
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from flask_cors import CORS
app = Flask(__name__)
CORS(app) 

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1") # Load BioBERT tokenizer
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1") # Load BioBERT model
df = pd.read_pickle('dataset_final.pkl') # Load preprocessed dataset
global_filtered_df = None # Store filtered DataFrame for deep search
global_last_recommendations = None # Store last recommendations for deep search
app = Flask(__name__) # Initialize Flask app

def fuzzy_field_score(str1, str2):
    if not str1 or not str2: # If either string is empty, return 0
        return 0
    return fuzz.token_set_ratio(str1, str2) / 100  # Normalize to [0,1]

def weighted_fuzzy_jaccard(paper_picos, query_picos, field_weights): # Calculate weighted fuzzy Jaccard score
    total = 0
    for k, w in field_weights.items(): # Iterate over each field and its weight
        score = fuzzy_field_score(paper_picos.get(k, ""), query_picos.get(k, "")) # Get fuzzy score for the field
        total += w * score # Normalize by weight
    return total

def get_embedding(text, max_length=512):
    # Tokenize and encode the text, truncating to max_length
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy() # Get mean of last hidden state as embedding

def normalize_scores(series):
    min_val = series.min() # Get minimum value
    max_val = series.max() # Get maximum value
    if max_val == min_val: # If all values are the same, return a constant series
        return pd.Series([0.5]*len(series), index=series.index)  # Avoid division by zero
    return (series - min_val) / (max_val - min_val) # Normalize to [0,1]

def find_optimal_k_elbow(X, k_min=2, k_max=15):
    """
    Automatically find the optimal number of clusters using the elbow (kneedle) method.
    Returns the optimal k and optionally plots the elbow curve with the straight line.
    """
    ks = list(range(k_min, k_max + 1)) # Range of k values to test
    inertias = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42) # Initialize KMeans with k clusters
        kmeans.fit(X) # Fit the model to the data
        inertias.append(kmeans.inertia_) # Store inertia (sum of squared distances to closest cluster center)

    # Line from first to last point
    point1 = np.array([ks[0], inertias[0]]) # First point (k_min, inertia)
    point2 = np.array([ks[-1], inertias[-1]]) # Last point (k_max, inertia)
    line_vec = point2 - point1 # Vector from first to last point
    line_vec_norm = line_vec / np.linalg.norm(line_vec) # Normalize the vector

    distances = []
    for i in range(len(ks)):
        point = np.array([ks[i], inertias[i]]) # Current point (k, inertia)
        vec = point - point1 # Vector from first point to current point
        proj = np.dot(vec, line_vec_norm) * line_vec_norm # Project current point onto the line
        dist = np.linalg.norm(vec - proj) # Distance from current point to the line
        distances.append(dist) # Store the distance
    # Find the k with the maximum distance from the line
    optimal_k = ks[np.argmax(distances)] 
    return optimal_k

def recommend_papers(
    df, user_query_text, user_query_picos, top_n_cluster=3, top_n_diverse=2, combine_weight=0.5, q=0.5):
    # 1. Filter by disease
    disease = user_query_picos["D"] # Extract disease from user query
    disease_df = df[df['disease'] == disease].copy() # Ensure we work with a copy
    global global_filtered_df # Store for deep search
    # 2. Fuzzy matching on PICOS using each paper's picos_weights
    disease_df['weighted_fuzzy_score'] = disease_df.apply( # Apply fuzzy matching to each row
        lambda row: weighted_fuzzy_jaccard(
            json.loads(row['picos_json']) if isinstance(row['picos_json'], str) else row['picos_json'], # Parse picos_json
            user_query_picos,
            row['picos_weights'] # Get weights for the paper
        ),
        axis=1 # Apply along rows
    )

    # 3. Filter by quantile (top 50% by default)
    quantile_value = disease_df['weighted_fuzzy_score'].quantile(q) # Get the q-th quantile value
    filtered_df = disease_df[disease_df['weighted_fuzzy_score'] >= quantile_value].copy() # Filter papers above the quantile
    filtered_df = filtered_df.sort_values('weighted_fuzzy_score', ascending=False) # Sort by fuzzy score

    # 4. Clustering
    X = np.vstack(filtered_df['embedding'].values) # Convert embeddings to a 2D array
    optimal_k = find_optimal_k_elbow(X, k_min=2, k_max=15) # Find optimal number of clusters using elbow method
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42) # Initialize KMeans with optimal k
    filtered_df['cluster'] = kmeans.fit_predict(X) # Fit KMeans and assign cluster labels
    global_filtered_df = filtered_df.copy()  # Store for deep search
    # 5. Cosine similarity with user query
    user_emb = get_embedding(user_query_text) # Get embedding for user query text
    # Calculate cosine similarity between user query and paper embeddings
    filtered_df['cosine_similarity'] = cosine_similarity([user_emb], np.vstack(filtered_df['embedding'].values))[0]

    # 6. Normalize scores and combine
    filtered_df['norm_fuzzy'] = normalize_scores(filtered_df['weighted_fuzzy_score']) # Normalize fuzzy scores
    filtered_df['norm_cosine'] = normalize_scores(filtered_df['cosine_similarity']) # Normalize cosine similarity scores
    filtered_df['combined_score'] = combine_weight * filtered_df['norm_cosine'] + (1 - combine_weight) * filtered_df['norm_fuzzy'] # Combine scores with specified weight

    # 7. Get user's cluster
    user_cluster = kmeans.predict(user_emb.reshape(1, -1))[0] # Predict cluster for user query embedding
    user_cluster_papers = filtered_df[filtered_df['cluster'] == user_cluster].sort_values('combined_score', ascending=False) # Sort user's cluster papers by combined score
    top_cluster = user_cluster_papers.head(top_n_cluster) # Get top papers from user's cluster

    # 8. Find 2 closest clusters (excluding user's)
    centroids = kmeans.cluster_centers_ # Get cluster centroids
    user_centroid = centroids[user_cluster] # Get centroid of user's cluster
    distances = cdist([user_centroid], centroids, metric='euclidean')[0] # Calculate distances from user's centroid to all centroids
    close_clusters = np.argsort(distances) # Get indices of closest clusters
    close_clusters = [c for c in close_clusters if c != user_cluster][:top_n_diverse] # Exclude user's cluster and take top_n_diverse closest clusters

    # 9. Get top paper from each close cluster
    diverse = []
    for c in close_clusters: # Iterate over each close cluster
        cluster_papers = filtered_df[filtered_df['cluster'] == c].sort_values('combined_score', ascending=False) # Sort papers in the cluster by combined score
        if not cluster_papers.empty: # If the cluster has papers
            diverse.append(cluster_papers.iloc[0]) # Append the top paper from the cluster to diverse list

    # 10. Combine recommendations
    diverse_df = pd.DataFrame(diverse) # Convert diverse list to DataFrame
    recommendations = pd.concat([top_cluster, diverse_df], axis=0, ignore_index=True) # Combine top cluster and diverse papers
    
    # 11. Prepare output JSON
    output = []
    for _, row in recommendations.iterrows(): # Iterate over each recommended paper
        output.append({
            "title": row.get("title", ""),
            "abstract": row.get("text", ""),
            "picos_json": row.get("picos_json", ""),
            "weighted_fuzzy_score": round(row.get("weighted_fuzzy_score", 0),4),
            "cosine_similarity": round(row.get("cosine_similarity", 0),2),
            "cluster": int(row.get("cluster", -1)),
            "user_cluster": int(user_cluster),
            "user_query_picos": user_query_picos,
            "user_query_text": user_query_text
        })
    return output

def deep_search_papers(filtered_df, selected_recommendation):
    cluster_id = selected_recommendation['cluster'] # Get the cluster ID of the selected recommendation
    cluster_papers = filtered_df[filtered_df["cluster"] == cluster_id].copy() # Filter papers in the same cluster
    cluster_papers = cluster_papers[cluster_papers["title"] != selected_recommendation["title"]] # Exclude the selected recommendation from the cluster papers
    selected_emb = get_embedding(selected_recommendation["abstract"]) # Get embedding for the selected recommendation's abstract
    
    # Ensure user_query_picos is a dict
    user_query_picos = selected_recommendation["user_query_picos"] # Extract user query PICOS from the selected recommendation
    if isinstance(user_query_picos, str): # If it's a string, parse it as JSON
        user_query_picos = json.loads(user_query_picos) # Ensure it's a dictionary
    
    cluster_papers["cosine_similarity"] = cluster_papers["abstract"].apply(
        lambda x: cosine_similarity([selected_emb], [get_embedding(x)])[0][0] # Calculate cosine similarity
    )
    cluster_papers["fuzzy_score"] = cluster_papers["picos_json"].apply(
        lambda pj: weighted_fuzzy_jaccard(
            json.loads(pj) if isinstance(pj, str) else pj, # Parse picos_json
            user_query_picos,
            selected_recommendation.get("picos_weights", {}) # Use weights from the selected recommendation
        )
    )
    # Normalize scores and combine
    cluster_papers["combined_score"] = 0.5 * cluster_papers["cosine_similarity"] + 0.5 * cluster_papers["fuzzy_score"]
    # Sort by combined score and take top 2
    top_similar = cluster_papers.sort_values("combined_score", ascending=False).head(2)
    
    result = []
    result.append({ # Add selected recommendation to the result
        "title": selected_recommendation["title"],
        "abstract": selected_recommendation["abstract"]
    })
    for _, row in top_similar.iterrows():
        result.append({
            "title": row["title"],
            "abstract": row["abstract"] if "abstract" in row else row.get("text", ""),
            "cosine_similarity": round(row["cosine_similarity"], 2),
            "fuzzy_score": round(row["fuzzy_score"], 4)
        })
    return result

# Route to handle recommendation requests
@app.route('/recommend', methods=['POST'])
def recommend(): # Handle recommendation requests
    global global_last_recommendations # Store last recommendations for deep search
    global global_filtered_df # Store filtered DataFrame for deep search
    data = request.json # Get JSON data from the request
    print("Raw input:", data)  # Debug print
    if "query_text" in data and "picos_json" in data: # Check if it's a recommendation request
        # Recommend intent
        user_query_text = data.get("query_text") # Get user query text
        user_query_picos = json.loads(data.get("picos_json")) # Parse user query PICOS JSON
        print("user_query_text:", user_query_text) # Debug print
        print("user_query_picos:", user_query_picos) # Debug print
        recommendations = recommend_papers(
            df,
            user_query_text,
            user_query_picos,
            top_n_cluster=1,
            top_n_diverse=2,
            combine_weight=0.5
        )
        global_last_recommendations = recommendations # Store last recommendations for deep search
        return jsonify({"recommendations": recommendations}) # Return recommendations as JSON
    elif "paper_index" in data: # Check if it's a deep search request
        paper_index = int(data["paper_index"]) - 1 # Convert to zero-based index
        recommendations = global_last_recommendations # Get last recommendations
        prev_rec = recommendations[paper_index] # Get the selected recommendation
        filtered_df = global_filtered_df # Filtered DataFrame from the last recommendation
        deep_results = deep_search_papers(filtered_df, prev_rec) # Perform deep search based on the selected recommendation
        return jsonify({"deep_recommendations": deep_results})# Return deep search results as JSON


if __name__ == '__main__': # Main entry point for the Flask app
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=5555) # Run the Flask app on all interfaces at port 5000