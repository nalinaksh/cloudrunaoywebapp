from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')

corpus = ["What are the three kinds of suffering?",
"What was the task given to mankind?",
"What is Maya?",
"What is the importance of having a Guru?",
"Can a master transfer his realization of cosmic consciousness?",
"What is Yoga?",
"What is the Eightfold path to Yoga?",
"What is Kriya Yoga?",
"Why is Kriya called the airplane route to God?",
"What is Pranayama?",
"What is the threefold nature of God?",
"What is the greatest proof of the existence of God?",
"Why shall we give God first place in our life?",
"How to deal with desires?",
"What role does fate play?",
"How powerful are thoughts?",
"What is cosmic consciousness?",
"What is AUM?",
"What is the spiritual eye?",
"What is the first commandment?",
"How to attain bliss?",
"How to overcome suffering?",
"How to make difficult decisions in life?",
"What does a soul reincarnate?",
"How to get out of the cycle of reincarnation?",
"How can we trancend the mind and body?",
"How to overcome or destroy Karma?",
"How to achieve even mindedness?",
"How to have a balanced life",
"How to cultivate devotion to God?",
"How important is the role of self effort in spirituality?",
"How to achieve attunement with Divine Will?",
"Does God takes favours?",
"Is there any shortcut to liberation?",
"Why God is hiding and does not reveal Himself?",
"How practical is it to seek God?"]

corpus_embeddings = embedder.encode(corpus)

def recommend(query):
  query_embedding = embedder.encode([query])
  hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
  hits = hits[0]      #Get the hits for the first query
  questions = []
  for hit in hits:
    questions.append(corpus[hit['corpus_id']])
  return questions
