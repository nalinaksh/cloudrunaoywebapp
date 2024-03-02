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
"Is it conceivable to give God second place in life?",
"How does a desire arise?",
"What role does fate play?",
"How powerful are thoughts?",
"What is cosmic consciousness?",
"What is AUM?",
"What is the spiritual eye?",
"what is Kutastha?",
"What is the first commandment?",
"How to attain bliss?",
"How to overcome suffering?",
"Why does a soul reincarnate?",
"Why reincarnation?",
"How to get out of the cycle of reincarnation?",
"How can we transcend the mind and body?",
"How to overcome or destroy Karma?",
"How to achieve even mindedness?",
"How to live a balanced life?",
"How to cultivate devotion to God?",
"How important is the role of self effort in spirituality?",
"How to achieve attunement with Divine Will?",
"Does God takes favours?",
"Is there any shortcut to liberation?",
"Why God is hiding and does not reveal Himself?",
"How to get rid of fear?",
"How to be happy?",
"How to solve life's problems?",
"How to live in harmony with Nature?",
"What is Sat-Chit-Ananda",
"What are the benefits of meditation?",
"What is twofold proof of God in meditation?",
"Are we divine beings?",
"What is Nirbikalpa Samadhi?",
"How to have peace?",
"What is Dharma?",
"What makes us act against our best interests?",
"Does a Soul ever be born or die?",
"What happens after death?",
"Tell me about Jivanmukta, Siddha and Paramukta?",
"How practical is it to seek God?"]

corpus_embeddings = embedder.encode(corpus)

def recommend(query):
  query_embedding = embedder.encode([query])
  hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=6)
  hits = hits[0]      #Get the hits for the first query
  questions = []
  for hit in hits:
    questions.append(corpus[hit['corpus_id']])
  return questions
