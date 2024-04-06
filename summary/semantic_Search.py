from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
print("line 1")
df = pd.read_csv("/home/gautam/Documents/medium_articles.csv")
print("line------ 1")

# print(len(df)) # 192368
# print("line number 2")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("line 000001")

print("line number 3")
# Concatenate titles and article texts
corpus = df["title"].head(10) + ". " + df["text"].head(10)

# Create embeddings from titles and texts of articles
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, )

# Convert tensor to NumPy array
corpus_embeddings_np = corpus_embeddings.numpy()

print("line number 4")
# # Specify the file path
file_path = "/home/gautam/Documents/wspace/video_Search/summary.npy"
print("line number 5")

# # Save the embeddings
# np.save(file_path, corpus_embeddings_np)

print("line number 6")
corpus_embeddings_np = np.load(file_path)

print("line number 7")
query = 'dearm purpose'
top_k = 10


# Find the closest top_k sentences of the corpus based on cosine similarity
query_embedding = embedder.encode(query, convert_to_tensor=True)

print("-----------query---------")

hits = util.semantic_search(query_embedding, corpus_embeddings_np, top_k=top_k)
print("-----------query@@@@@@@@@@@@@@@@@@@@@---------")

hits = hits[0] # Get the hits for the first query


print(f"\nTop {top_k} most similar sentences in corpus:")
for hit in hits:
    hit_id = hit['corpus_id']
    article_data = df.iloc[hit_id]
    title = article_data["title"]
    print(hit_id)
    print("url of medium is", df.loc[hit_id, "url"])
    print("-", title, "(Score: {:.4f})".format(hit['score']))
