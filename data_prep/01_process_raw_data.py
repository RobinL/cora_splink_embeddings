# from https://www.cs.utexas.edu/users/ml/riddle/data.html
import duckdb
import numpy as np
import pandas as pd
from openai import OpenAI

from my_secrets import OPENAI_API_KEY

df = pd.read_csv("cora_corrected.csv")
pd.options.display.max_colwidth = 100
df_10_rows = df.tail(10).copy()
# display(df_10_rows)
client = OpenAI(api_key=OPENAI_API_KEY)


def batch_embeddings(
    df, col, model="text-embedding-3-small", dimensions=100, batch_size=500
):
    # Initialize a list to hold the embeddings
    df[f"{col}_embeddings"] = np.nan
    df[f"{col}_embeddings"] = df[f"{col}_embeddings"].astype(object)

    # Function to fetch embeddings for a batch of texts
    def get_embeddings(texts, model="text-embedding-3-small", dimensions=100):
        try:
            response = client.embeddings.create(
                input=texts, model=model, dimensions=dimensions
            )
            # Assuming 'response' is an object with a 'data' attribute that is a list of Embedding objects
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

    # Iterate over the DataFrame in batches
    for start_idx in range(0, len(df), batch_size):
        print(f"Processing batch starting at index {start_idx}")
        # End index for the current batch
        end_idx = start_idx + batch_size
        # Extract the batch of texts, filtering out NaNs and keeping track of their original indices
        batch_texts = df[col][start_idx:end_idx]
        non_null_indices = batch_texts.dropna().index
        non_null_texts = batch_texts.dropna().tolist()

        # Fetch embeddings for the non-null texts
        if non_null_texts:  # Check if the list is not empty
            batch_embeddings = get_embeddings(
                non_null_texts,
            )
            # Insert embeddings back into the correct positions in the 'embeddings' list
            for idx, embedding in zip(non_null_indices, batch_embeddings):
                df.at[idx, f"{col}_embeddings"] = list(embedding)


# Example usage
# for col in ["author", "title", "venue", "note"]:
# for col in ["note"]:
#     print(f"Processing column: {col}")
#     batch_embeddings(df_10_rows, col, model="text-embedding-3-small")
# df_10_rows

for col in ["author", "title", "venue", "note"]:
    batch_embeddings(df, col, model="text-embedding-3-small")

df["unique_id"] = df.reset_index().index

df.to_parquet("cora_corrected_all_with_embeddings.parquet")
