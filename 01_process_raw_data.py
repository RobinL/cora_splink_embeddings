# from https://www.cs.utexas.edu/users/ml/riddle/data.html
import duckdb
import numpy as np
import pandas as pd
from openai import OpenAI

from my_secrets import OPENAI_API_KEY

# read arff_data_section.csv in pandas where all data is in "" and separated by commas
df = pd.read_csv("cora_corrected.csv")
pd.options.display.max_colwidth = 100
df_3_rows = df.tail(3).copy()

client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text, model="text-embedding-3-small"):

    if not isinstance(text, str):
        print(text)
        return None
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=model, dimensions=100)
        .data[0]
        .embedding
    )


res1 = client.embeddings.create(
    input=[np.nan],
    model="text-embedding-3-small",
    dimensions=100,
)
res1.data[0].embedding
for col in ["author", "title", "venue", "note"]:
    print("starting", col)
    df[f"{col}_embedding"] = df[f"{col}"].apply(
        lambda x: get_embedding(x, model="text-embedding-3-small")
    )
df.to_parquet("cora_corrected_all_with_embeddings.parquet")


df
# author, title, venue, note, pages

# for col in ["author", "title", "venue", "note"]:
#     df_3_rows[f"{col}_embedding"] = df_3_rows[f"{col}"].apply(
#         lambda x: get_embedding(x, model="text-embedding-3-small")
#     )
# df_3_rows.to_parquet("cora_corrected_3_rows_with_embeddings.parquet")

# import numpy as np
# from numpy.linalg import norm
# e1 = df_3_rows["title_embedding"].iloc[1]
# e2 = df_3_rows["title_embedding"].iloc[2]
# dot_product = np.dot(e1, e2)
# norm_vec1 = np.linalg.norm(e1)
# norm_vec2 = np.linalg.norm(e2)
# similarity = dot_product / (norm_vec1 * norm_vec2)
# similarity


# res1 = client.embeddings.create(
#     input=["avrim blum, merrick furst, michael kearns, and richard j. lipton."],
#     model="text-embedding-3-small",
#     dimensions=100,
# )
# res2 = client.embeddings.create(
#     input=["a. blum, m. furst, m. j. kearns, and richard j. lipton."],
#     model="text-embedding-3-small",
#     dimensions=100,
# )


# e1 = res1.data[0].embedding
# e2 = res2.data[0].embedding
