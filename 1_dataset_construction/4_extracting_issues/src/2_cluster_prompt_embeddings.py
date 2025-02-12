import torch
import umap
import hdbscan
import logging
import pandas as pd
import numpy as np
import fire
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


def c_tf_idf(prompts, m, ngram_range=(1, 2)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(prompts)
    t = count.transform(prompts).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_cluster(tf_idf, count, prompts_by_cluster, top_n_words):
    words = count.get_feature_names_out()
    labels = list(prompts_by_cluster["cluster_id"])
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -top_n_words:]
    top_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_words


def reduce_dim_umap(embeddings, umap_dim, umap_min_dist, seed, umap_n_neighbors, umap_metric):
    
    umap_embeddings = umap.UMAP(
        random_state = seed,
        n_neighbors = umap_n_neighbors, 
        n_components = umap_dim, 
        min_dist = umap_min_dist,
        metric = umap_metric).fit_transform(embeddings)
    
    return umap_embeddings

def reduce_dim_pca(embeddings, pca_dim):
    pca = PCA(n_components=pca_dim)
    pca_embeddings = pca.fit_transform(embeddings)

    return pca_embeddings

def get_sourceprop_in_cluster(cluster_id, source, prompts_df):
    n_total = len(prompts_df[prompts_df.cluster_id == cluster_id])
    n_source = len(prompts_df[(prompts_df.cluster_id == cluster_id) & (prompts_df.source == source)])
    return n_source / n_total


def dominated_by(row, threshold):
    sources = ["lmsys", "sharegpt", "hhonline", "prism", "wildchat"]
    max_source = max(sources, key=lambda source: row[f"prop_{source}"])
    if row[f"prop_{max_source}"] >= threshold:
        return max_source
    

def main(
        # input paths
        prompts_input_path: str,
        prompts_input_col: str,
        embeddings_input_path: str,
        
        # output paths
        output_path_prompts: str,
        output_path_clusters: str,

        # dimensionality reduction parameters
        dimensionality_reduction_method: str,

        # umap parameters
        umap_dim: int,
        umap_min_dist: float,
        umap_n_neighbors: int, 
        umap_metric: str,

        # pca parameters
        pca_dim: int,

        # hdbscan parameters
        hdb_min_cluster_size: int,
        hdb_min_samples: int,
        hdb_metric: str,
        hdb_cluster_selection_method: str,
        hdb_epsilon: float,
        
        # top n parameters for clusters
        top_n_words: int,
        top_n_prompts: int,
        random_n_prompts: int,
        
        # other parameters
        log_level: str,
        seed: int,
        compute_embeddings_for_visualisation: bool = False
        ):

    # set up logging
    logging.basicConfig(level=getattr(logging, log_level.upper()), format='%(asctime)s %(levelname)s %(message)s')

    # Load prompts
    prompt_texts = list(pd.read_csv(prompts_input_path)[prompts_input_col])
    prompt_ids = list(pd.read_csv(prompts_input_path)["id"])
    logging.info(f"Loaded {len(prompt_texts)} prompt texts and IDs.")
    
    # Load embeddings
    prompt_embeddings = torch.load(embeddings_input_path)
    logging.info(f"Loaded embeddings with shape {prompt_embeddings.shape}.")

    # Reduce dimensionality of embeddings 

    if dimensionality_reduction_method == "umap":
        reduced_embeddings = reduce_dim_umap(prompt_embeddings, umap_dim, umap_min_dist, seed, umap_n_neighbors, umap_metric)
        logging.info(f"Reduced embedding dimensionality to {umap_dim} with UMAP.")

    elif dimensionality_reduction_method == "pca":
        reduced_embeddings = reduce_dim_pca(prompt_embeddings, pca_dim)
        logging.info(f"Reduced embedding dimensionality to {pca_dim} with PCA.")
        
    # Create clusters with HDBScan
    cluster = hdbscan.HDBSCAN(
        min_cluster_size = hdb_min_cluster_size,
        min_samples = hdb_min_samples,
        metric = hdb_metric,
        cluster_selection_epsilon = hdb_epsilon,                      
        cluster_selection_method = hdb_cluster_selection_method).fit(reduced_embeddings)
    logging.info(f"Created {len(set(cluster.labels_))} clusters with a minimum size of {hdb_min_cluster_size} prompts with HDBScan.")

    # Create dataframe where prompts are labelled with cluster_ids and reduced embeddings
    reduced_embeddings_list = []
    for row in reduced_embeddings:
        reduced_embeddings_list.append(row.tolist())

    prompt_df = pd.DataFrame({'id': prompt_ids, prompts_input_col: prompt_texts, 'cluster_id': cluster.labels_, 'reduced_embedding': reduced_embeddings_list})
    logging.info(f"Created df where prompts are labelled with cluster_ids and umap_embedding, with shape {prompt_df.shape}.")

    # Compute centroids of each cluster as average of reduced embeddings
    cluster_centroids = []
    for i in range(len(set(cluster.labels_))):
        cluster_centroids.append(np.mean(reduced_embeddings[cluster.labels_ == i], axis=0).tolist())

    # compute distance of each prompt to its cluster centroid
    cluster_distances = []
    for i in range(len(reduced_embeddings)):
        cluster_distances.append(np.linalg.norm(reduced_embeddings[i] - cluster_centroids[cluster.labels_[i]]))

    # add distance to centroid to output dataframe
    prompt_df['distance_to_centroid'] = cluster_distances
    logging.info(f"Added distance to cluster centroid to prompt df.")

    if compute_embeddings_for_visualisation:
        # Create 3d and 2d embeddings for visualisation
        umap_embeddings_3d = reduce_dim_umap(prompt_embeddings, 3, umap_min_dist, seed, umap_n_neighbors)
        result_3d = pd.DataFrame(umap_embeddings_3d, columns=['x3', 'y3', 'z3'])
        umap_embeddings_2d = reduce_dim_umap(prompt_embeddings, 2, umap_min_dist, seed, umap_n_neighbors)
        result_2d = pd.DataFrame(umap_embeddings_2d, columns=['x2', 'y2'])
        vis_embeddings = pd.concat([result_2d, result_3d], axis = 1)
        prompt_df = pd.concat([prompt_df, vis_embeddings], axis = 1)
        logging.info(f"Created 3d and 2d embeddings for visualisation, and concatenated with prompt_df")

    logging.info(f"Final shape of prompt_df is {prompt_df.shape}.")

    # Export to csv
    prompt_df.to_csv(output_path_prompts, index=False)
    logging.info(f"Saved prompt-level results to {output_path_prompts}.")
    
    # Create df where every cluster has all prompts within that cluster concatenated into a single string
    prompts_by_cluster = prompt_df.groupby(['cluster_id'], as_index = False).agg({prompts_input_col: ' '.join})

    # Run tf-idf, then use that to identify top 20 uni/bigrams for each cluster
    tf_idf, count = c_tf_idf(prompts_by_cluster[prompts_input_col].values, m=len(prompt_texts))
    top_words = extract_top_n_words_per_cluster(tf_idf, count, prompts_by_cluster, top_n_words)
    
    # Create cluster_df of cluster IDs with size of cluster
    cluster_df = pd.DataFrame(prompt_df.cluster_id.value_counts())
    cluster_df.reset_index(inplace=True)
    cluster_df.columns = ["cluster_id", "cluster_size"]

    # Write top n prompts closest to centroid to column in cluster_df
    centroid_prompts = prompt_df.groupby('cluster_id').apply(lambda x: x.nsmallest(top_n_prompts, 'distance_to_centroid')).reset_index(drop=True)
    centroid_prompts = centroid_prompts.groupby('cluster_id').agg({prompts_input_col: lambda x: list(x)}).reset_index()
    centroid_prompts.rename(columns={prompts_input_col: 'top_prompts'}, inplace=True)
    cluster_df = cluster_df.merge(centroid_prompts, on='cluster_id', how='left')

    # Write n random prompts to column in cluster_df
    random_prompts = prompt_df.groupby('cluster_id').apply(lambda x: x.sample(random_n_prompts)).reset_index(drop=True)
    random_prompts = random_prompts.groupby('cluster_id', ).agg({prompts_input_col: lambda x: list(x)}).reset_index()
    random_prompts.rename(columns={prompts_input_col: 'random_prompts'}, inplace=True)
    cluster_df = cluster_df.merge(random_prompts, on='cluster_id', how='left')

    # Write top n words to columns in cluster_df df    
    cluster_df['top_words'] = cluster_df.cluster_id.apply(lambda x: [pair[0] for pair in top_words[x]])

    # Write proportion of prompts from each source to columns in cluster_df
    prompt_df["source"] = prompt_df["id"].apply(lambda x: x.split("-")[0])
    for source in ["lmsys", "sharegpt", "hhonline", "prism", "wildchat"]:
        cluster_df[f"prop_{source}"] = cluster_df.cluster_id.apply(lambda x: get_sourceprop_in_cluster(x, source, prompt_df))
    cluster_df["dominated_by"] = cluster_df.apply(lambda row: dominated_by(row, 0.8), axis=1)

    # Export cluster_df to csv
    cluster_df[["cluster_id","cluster_size","dominated_by","prop_wildchat","prop_lmsys","prop_sharegpt","prop_hhonline","prop_prism","top_words","top_prompts","random_prompts"]].to_csv(output_path_clusters, index=False)
    logging.info(f"Saved cluster-level results to {output_path_clusters}.")

    return


if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    et = time.time()
    logging.info(f'Execution time: {et - st:.2f} seconds\n')