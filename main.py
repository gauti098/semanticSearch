###semantic search with elastic embeddings
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests


class VideoSearchEngine:
    def __init__(self, index_name="videos", host="localhost", port=9200):
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        self.index_name = index_name
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def setup_index(self):
        if not self.es.indices.exists(index=self.index_name):
            mappings = {
                "properties": {
                    "title": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 384},
                    "url": {"type": "keyword"}
                }
            }

            self.es.indices.create(index=self.index_name, body={"mappings": mappings})
            print(f"Index '{self.index_name}' created.")
        else:
            print(f"Index '{self.index_name}' already exists.")



    def index_article(self, videos):
        for video in videos:
            title = video.get('title', '')
            url = video.get('videoUrl', '')

            if not title:
                print("Skipping video indexing due to missing title.")
                continue

            # # Generate embedding for the title
            embedding = self.embedder.encode(title)

            # Index the video with its embedding
            doc = {"title": title, "embedding": embedding.tolist(), "url": url}
            self.es.index(index=self.index_name, body=doc)
            print(f"Indexed video: {title}")
        # title = "nsl demo video 12th march"
        # embedding = self.embedder.encode(title)

        # # Index the video with its embedding
        # doc = {"title": title, "embedding": embedding.tolist(), "url": "https://video.com"}
        # self.es.index(index=self.index_name, body=doc)



    # def semantic_search(self, query, top_k=10):
    #     query_embedding = self.embedder.encode(query)
    #     search_body = {
    #         "query": {
    #             "script_score": {
    #                 "query": {"match_all": {}},
    #                 "script": {
    #                     "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
    #                     "params": {"query_vector": query_embedding.tolist()}
    #                 }
    #             }
    #         }
    #     }
    #     response = self.es.search(index=self.index_name, body=search_body, size=top_k)
    #     hits = response["hits"]["hits"]
    #     return [{"title": hit["_source"]["title"], "url": hit["_source"]["url"]} for hit in hits]
    def semantic_search(self, query, top_k=15):
        query_embedding = self.embedder.encode(query)
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
        response = self.es.search(index=self.index_name, body=search_body, size=top_k)
        hits = response["hits"]["hits"]
        max_score = response["hits"]["max_score"]  # Get the maximum score in the response

        # Calculate the score percentage for each hit
        results = []
        for hit in hits:
            score = hit["_score"]
            score_percentage = (score / max_score) * 100  # Calculate the percentage
            results.append({
                "title": hit["_source"]["title"],
                "url": hit["_source"]["url"],
                "score_percentage": score_percentage  # Include score percentage in the result
            })
        return results


# seting index
search_engine = VideoSearchEngine()
search_engine.setup_index()


# Fetch videos
videos = search_engine.fetch_data_from_api()
# print(videos)


# Index videos embeddings
search_engine.index_article(videos)


# Performing semantic search based on query
query = "rcb punjab ipl match"
results = search_engine.semantic_search(query)

# Display top results
print(f"Top {len(results)} videos related to '{query}':")
for idx, result in enumerate(results, start=1):
    print(f"{idx}. {result['title']}: {result['url']}: {result['score_percentage']}")