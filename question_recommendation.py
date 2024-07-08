from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')

corpus = ["What are the three kinds of suffering?",
"What was the task given to mankind?",
"What is Maya?",
"What is Yoga?",
"What is the Eightfold path to Yoga?",
"What is Kriya Yoga?",
"Why is Kriya Yoga called the airplane route?",
"What is Pranayama?",
"What is the threefold nature of God?",
"What is Ego?",
"Is desire our greatest enemy?",
"How powerful are thoughts?",
"What is cosmic consciousness?",
"What is AUM?",
"What is the spiritual eye?",
"What is Kutastha?",
"What is the first commandment?",
"How to overcome suffering?",
"How to overcome or destroy Karma?",
"How to achieve even mindedness?",
"How to live a balanced life?",
"How to get rid of fear?",
"How to be happy?",
"How to solve life's problems?",
"How to live in harmony with Nature?",
"What is Sat-Chit-Ananda?",
"What are the benefits of meditation?",
"What is Nirbikalpa Samadhi?",
"How to have peace?",
"What is Dharma?",
"What makes us act against our best interests?",
"Tell me about Jivanmukta, Siddha and Paramukta!"]

corpus_embeddings = embedder.encode(corpus)

def recommend(query):
  query_embedding = embedder.encode([query])
  hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=4)
  hits = hits[0]      #Get the hits for the first query
  questions = []
  for hit in hits:
    questions.append(corpus[hit['corpus_id']])
  return questions
