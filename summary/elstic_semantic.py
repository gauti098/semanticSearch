from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("/home/gautam/Documents/medium_articles.csv")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("line number 22222222")
# Load embeddings
file_path = "/home/gautam/Documents/wspace/video_Search/summary.npy"
corpus_embeddings_np = np.load(file_path)

print("line number 222000000022222")

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
index_name = 'semanticsearch'

print("line number 2222-----------2222")

# Index embeddings into Elasticsearch
for i, embedding in enumerate(corpus_embeddings_np):
    doc = {
        'embedding': embedding.tolist(),
        'title': df.iloc[i]['title'],
        'url': df.iloc[i]['url']
    }
    print("line number-------------------")

    es.index(index=index_name, body=doc)

print("Embeddings indexed successfully.")

# Function to perform semantic search
def semantic_search(query_embedding, top_k=10):
    search_body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
    }

    response = es.search(index=index_name, body=search_body, size=top_k)
    hits = response['hits']['hits']
    return hits

# Perform semantic search
query = 'dearm purpose'
query_embedding = embedder.encode(query, convert_to_tensor=True)
hits = semantic_search(query_embedding)

# Print search results
print(f"\nTop {len(hits)} most similar sentences in corpus:")
for hit in hits:
    print("url of medium is", hit['_source']['url'])
    print("-", hit['_source']['title'], "(Score: {:.4f})".format(hit['_score']))
