from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

corpus = ["What are the three kinds of suffering?",
"What was the task given to mankind?",
"What is Maya?",
"Can a master transfer his realization of cosmic consciousness?",
"What is Yoga?",
"What is the Eightfold path to Yoga?",
"What is Kriya Yoga?",
"Why is Kriya called the airplane route to God?",
"What is the threefold nature of God?",
"What is the greatest proof of the existence of God?",
"Why shall we give God first place in our life?",
"How to deal with desires?",
"What is the astral world?",
"What is the causal world?",
"What role does fate play?",
"How powerful are thoughts?",
"What is the subconscious mind?",
"What is the superconscious mind?",
"What is AUM?",
"What is the spiritual eye?",
"What is the first commandment?"]

corpus_embeddings = embedder.encode(corpus)

def recommend(query):
  query_embedding = embedder.encode([query])
  hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
  hits = hits[0]      #Get the hits for the first query
  questions = []
  for hit in hits:
    questions.append(corpus[hit['corpus_id']])
  return questions
