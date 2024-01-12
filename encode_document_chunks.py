from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

corpus_embedding = model.encode(result_chunks, normalize_embeddings=True)

question = "What is Kriya Yoga?"
query_instruction = "Represent this sentence for searching relevant passages: "
query_embedding = model.encode([query_instruction + question])

print(util.semantic_search(query_embedding, corpus_embedding))
