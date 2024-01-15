from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'AOY_text_chunks.csv'

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

#access chunks
# print(df['Chunk Content'][100])

#load saved embeddings on a CPU to make model predictions
all_embeddings = np.load('AOY_Embeddings.npy')

#to download model files (downloaded folder - bge-large-en-v1.5)
# !git lfs install
# !git clone https://huggingface.co/BAAI/bge-large-en-v1.5

#move the model to a CPU
model = SentenceTransformer('BAAI/bge-large-en-v1.5').cpu()

def retrieve_answers(question):
  query_instruction = "Represent this sentence for searching relevant passages: "
  query_embedding = model.encode([query_instruction + question])
  result = util.semantic_search(query_embedding, all_embeddings, top_k=3)
  docs = []
  for i in range(3):
    corpus_id = result[0][i]['corpus_id']
    doc = df['Chunk Content'][corpus_id]
    docs.append(doc)
  return docs
