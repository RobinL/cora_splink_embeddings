import numpy as np
from openai import OpenAI

from my_secrets import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

res1 = client.embeddings.create(
    input=["splink is a tool for data linkage"],
    model="text-embedding-3-small",
    dimensions=100,
)
res2 = client.embeddings.create(
    input=["splink is software for data deduplication at scale "],
    model="text-embedding-3-small",
    dimensions=100,
)


e1 = res1.data[0].embedding
e2 = res2.data[0].embedding


dot_product = np.dot(e1, e2)
norm_vec1 = np.linalg.norm(e1)
norm_vec2 = np.linalg.norm(e2)
similarity = dot_product / (norm_vec1 * norm_vec2)
similarity
