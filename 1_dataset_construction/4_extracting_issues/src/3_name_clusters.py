import openai
from openai import OpenAI

import fire
import time
import pandas as pd

from retrying import retry
from decouple import config
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

tqdm.pandas()


class GPTWrapper:

    def __init__(self, gen_model):
        self.model_name = gen_model
        self.client = OpenAI(
            api_key=config('OPENAI_API_KEY'), # reads from a file called ".env" in root directory of repo
        )

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000) # 2^x * 1000 milliseconds between each retry, up to 10 seconds, then 10 seconds afterwards
    def name_cluster(self, top_prompts, random_prompts, top_words):

        prompt_template = f"Your task is to list up to three specific and distinct nouns or noun phrases to describe a cluster of prompts based on the following information.\n\n\
        Typical words used in the cluster are: {top_words}\n\n\
        Typical prompts in the cluster are: {top_prompts}\n\n\
        Other random prompts from the cluster are: {random_prompts}\n\n\
        Remember to use specific and distinct nouns or noun phrases to describe the cluster. Do not enumerate but rather separate the nouns or noun phrases by commas in one row. \n\n\
        Nouns:"

        input = [{"role": "system", "content": ""},
                 {"role": "user", "content": prompt_template.format(str(top_prompts), str(top_words))}]

        try:
            response = self.client.chat.completions.create(model = self.model_name,
                messages = input,
                temperature = 0,
                max_tokens = 256,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0,
                )

            return response.choices[0].message.content
        
        except openai.OpenAIError as e:
            print(f"OpenAIError: {e}. Retrying with exponential backoff.")
            raise e
    
    def name_clusters_in_parallel(self, top_prompts, random_prompts, top_words, max_workers):
        completions = thread_map(self.name_cluster, top_prompts, random_prompts, top_words, max_workers=max_workers)
        return [c for c in completions]


def clean_prompts(prompts):
    
    prompts = prompts.str.replace("\n", " ").str.replace("\r", " ").str.replace("\t", " ").str.replace("  ", " ")
    prompts = prompts.str.replace("{", "").str.replace("}", "")
    prompts = prompts.apply(lambda x: " ".join(x.split()))

    return prompts


def main(gen_model: str, input_path: str, output_path: str, num_samples: int = 0, max_workers: int = 1, seed: int = 123):
    
    # load csv
    cluster_df = pd.read_csv(input_path)
    print(f"Loaded cluster overview from {input_path}: {cluster_df.shape[0]} clusters")

    # optional: select random sample from df -- useful for debugging
    if num_samples > 0:
        cluster_df = cluster_df.sample(num_samples, random_state=seed)
        print(f"Sampled {num_samples} rows from data")

    # initialize GPTWrapper
    gpt = GPTWrapper(gen_model)
    print(f"Initialized OpenAI model: {gen_model}")

    # minimal preprocessing of top and random prompts to avoid API errors
    cluster_df["top_prompts"] = clean_prompts(cluster_df["top_prompts"])
    cluster_df["random_prompts"] = clean_prompts(cluster_df["random_prompts"])

    # write gpt completion to new column
    cluster_df["gpt_description"] = gpt.name_clusters_in_parallel(cluster_df.top_prompts, cluster_df.random_prompts, cluster_df.top_words, max_workers=max_workers)

    # write model name to column
    cluster_df["description_model"] = gen_model

    # reorder columns and drop unnecessary ones
    cluster_df = cluster_df[["cluster_id", "cluster_size", "gpt_description",
                             "dominated_by", "prop_wildchat", "prop_lmsys", "prop_sharegpt", "prop_hhonline", "prop_prism",
                             "top_words", "top_prompts", "random_prompts", "description_model"]]

    # sort by cluster size
    cluster_df = cluster_df.sort_values("cluster_size", ascending=False)

    # save to csv
    cluster_df.to_csv(output_path, index=False)

    return

if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    et = time.time()
    print(f'Execution time: {et - st:.2f} seconds')