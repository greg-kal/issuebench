import pandas as pd
import time
import torch
import fire
from sentence_transformers import SentenceTransformer


def main(input_path: str, num_samples: int, seed: int, input_col:str, embedding_model: str, batch_size: int, cache_dir: str, output_path: str):

    # load csv
    df = pd.read_csv(input_path)
    print(f"Loaded prompts from {input_path}: {df.shape[0]} rows")

    # optional: select random sample from dataframe -- useful for debugging
    if num_samples > 0:
        df = df.sample(num_samples, random_state=seed)
        print(f"Sampled {num_samples} rows from data")

    # load model for computing embeddings (sentence transformers also work for short paragraphs)
    model = SentenceTransformer(embedding_model, cache_folder=cache_dir)
    print(f"Loaded embedding model: {embedding_model}")

    # compute embeddings for each prompt
    embeddings = model.encode(list(df[input_col]), batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    print(f"Computed embeddings with shape {embeddings.shape}")

    # Save tensor
    torch.save(embeddings, output_path)
    print(f"Saved embeddings at {output_path}")

    return


if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    et = time.time()
    print(f'Execution time: {et - st:.2f} seconds\n')
