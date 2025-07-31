import subprocess
import fire
import time
import pandas as pd
import requests
import json
from retrying import retry
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tqdm.pandas()

class LlamaWrapper:
    def __init__(self, gen_model):
        self.model_name = gen_model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Test connection to Ollama
        self._test_connection()

    def _test_connection(self):
        """Test if Ollama is running and the model is available"""
        try:
            # Check if Ollama is running
            test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if test_response.status_code != 200:
                raise Exception(f"Ollama not running or not accessible: {test_response.status_code}")
            
            # Check if the model exists
            models = test_response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found in available models: {model_names}")
                logger.info("You may need to pull the model first: ollama pull " + self.model_name)
            else:
                logger.info(f"Successfully connected to Ollama with model {self.model_name}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Cannot connect to Ollama. Make sure it's running: {e}")

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=3)
    def name_cluster(self, top_prompts, random_prompts, top_words):
        """Generate cluster name using Llama via Ollama"""
        
        # Validate inputs
        if pd.isna(top_prompts) or pd.isna(random_prompts) or pd.isna(top_words):
            logger.warning("Found NaN values in input data, skipping cluster")
            return "Unknown Cluster"
        
        prompt_template = f"""Your task is to list up to three specific and distinct nouns or noun phrases to describe a cluster of prompts based on the following information.

Typical words used in the cluster are: {top_words}

Typical prompts in the cluster are: {top_prompts}

Other random prompts from the cluster are: {random_prompts}

Remember to use specific and distinct nouns or noun phrases to describe the cluster. Do not enumerate but rather separate the nouns or noun phrases by commas in one row.

Nouns:"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt_template,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 256
                    }
                },
                timeout=60  # 60 second timeout
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama HTTP error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            data = response.json()
            result = data.get("response", "").strip()
            
            if not result:
                logger.warning("Empty response from Ollama")
                return "Empty Response"
            
            logger.debug(f"Generated description: {result[:100]}...")
            return result
            
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            raise Exception("Ollama request timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise Exception("Invalid JSON response from Ollama")
    
    def name_clusters_in_parallel(self, top_prompts, random_prompts, top_words, max_workers):
        """Process clusters in parallel with error handling"""
        logger.info(f"Processing {len(top_prompts)} clusters with {max_workers} workers")
        
        try:
            # Use thread_map with proper error handling
            completions = thread_map(
                self.name_cluster, 
                top_prompts, 
                random_prompts, 
                top_words, 
                max_workers=max_workers,
                desc="Generating cluster descriptions"
            )
            
            # Log results
            successful = sum(1 for c in completions if c and c not in ["Unknown Cluster", "Empty Response"])
            logger.info(f"Successfully generated descriptions for {successful}/{len(completions)} clusters")
            
            return completions
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            completions = []
            for i, (tp, rp, tw) in enumerate(zip(top_prompts, random_prompts, top_words)):
                try:
                    result = self.name_cluster(tp, rp, tw)
                    completions.append(result)
                    if i % 10 == 0:
                        logger.info(f"Processed {i+1}/{len(top_prompts)} clusters")
                except Exception as cluster_error:
                    logger.error(f"Error processing cluster {i}: {cluster_error}")
                    completions.append("Error in Processing")
            
            return completions


def clean_prompts(prompts):
    """Clean and preprocess prompt text"""
    if prompts is None:
        return prompts
    
    # Handle NaN values
    prompts = prompts.fillna("")
    
    # Clean whitespace and special characters
    prompts = prompts.str.replace("\n", " ", regex=False)
    prompts = prompts.str.replace("\r", " ", regex=False)
    prompts = prompts.str.replace("\t", " ", regex=False)
    prompts = prompts.str.replace("  ", " ", regex=True)
    prompts = prompts.str.replace("{", "", regex=False)
    prompts = prompts.str.replace("}", "", regex=False)
    
    # Normalize whitespace
    prompts = prompts.apply(lambda x: " ".join(str(x).split()) if pd.notna(x) else "")
    
    return prompts


def main(gen_model: str = "llama3.1:70b",
         input_path: str = "all_clean_filtered_clusteroverview.csv",
         output_path: str = "all_clean_filtered_clusteroverview_named.csv",
         num_samples: int = 0,
         max_workers: int = 5,  # Reduced default to be more conservative
         seed: int = 123):
    
    logger.info("Starting cluster naming process")
    
    # Load CSV with error handling
    try:
        cluster_df = pd.read_csv(input_path)
        logger.info(f"Loaded cluster overview from {input_path}: {cluster_df.shape[0]} clusters")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return

    # Validate required columns
    required_columns = ["top_prompts", "random_prompts", "top_words"]
    missing_columns = [col for col in required_columns if col not in cluster_df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(cluster_df.columns)}")
        return

    # Optional: select random sample from df
    if num_samples > 0:
        if num_samples > len(cluster_df):
            logger.warning(f"Requested {num_samples} samples but only {len(cluster_df)} available")
            num_samples = len(cluster_df)
        cluster_df = cluster_df.sample(num_samples, random_state=seed)
        logger.info(f"Sampled {num_samples} rows from data")

    # Initialize LlamaWrapper
    try:
        llama = LlamaWrapper(gen_model)
    except Exception as e:
        logger.error(f"Failed to initialize Llama wrapper: {e}")
        return

    # Clean prompts
    logger.info("Cleaning prompt data")
    cluster_df["top_prompts"] = clean_prompts(cluster_df["top_prompts"])
    cluster_df["random_prompts"] = clean_prompts(cluster_df["random_prompts"])
    
    # Ensure top_words is clean too
    if "top_words" in cluster_df.columns:
        cluster_df["top_words"] = cluster_df["top_words"].fillna("")

    # Generate descriptions
    logger.info("Generating cluster descriptions")
    try:
        descriptions = llama.name_clusters_in_parallel(
            cluster_df["top_prompts"].tolist(), 
            cluster_df["random_prompts"].tolist(), 
            cluster_df["top_words"].tolist(), 
            max_workers=max_workers
        )
        
        cluster_df["llama_description"] = descriptions
        cluster_df["description_model"] = gen_model
        
        # Log statistics
        non_empty_descriptions = sum(1 for d in descriptions if d and d.strip() and d not in ["Unknown Cluster", "Empty Response", "Error in Processing"])
        logger.info(f"Generated {non_empty_descriptions} non-empty descriptions out of {len(descriptions)} total")
        
    except Exception as e:
        logger.error(f"Error generating descriptions: {e}")
        return

    # Reorder columns (only include columns that exist)
    base_columns = ["cluster_id", "cluster_size", "llama_description", "description_model"]
    optional_columns = ["dominated_by", "prop_wildchat", "prop_lmsys", "prop_sharegpt", "prop_hhonline", "prop_prism"]
    data_columns = ["top_words", "top_prompts", "random_prompts"]
    
    # Only include columns that exist in the dataframe
    final_columns = []
    for col in base_columns + optional_columns + data_columns:
        if col in cluster_df.columns:
            final_columns.append(col)
    
    cluster_df = cluster_df[final_columns]

    # Sort by cluster size if column exists
    if "cluster_size" in cluster_df.columns:
        cluster_df = cluster_df.sort_values("cluster_size", ascending=False)

    # Save to CSV
    try:
        cluster_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved results to {output_path}")
        
        # Verify the file was written correctly
        verification_df = pd.read_csv(output_path)
        if "llama_description" in verification_df.columns:
            non_empty_count = verification_df["llama_description"].notna().sum()
            logger.info(f"Verification: {non_empty_count} non-null descriptions in saved file")
        else:
            logger.error("Verification failed: llama_description column not found in saved file")
            
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        return

    logger.info("Cluster naming process completed successfully")


if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    et = time.time()
    logger.info(f'Execution time: {et - st:.2f} seconds')