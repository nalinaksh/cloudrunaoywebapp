from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the path to your CSV file
aoy_file_path = 'data/AOY/AOY.csv'
gita_file_path = 'data/Gita/Bhagwad_Gita_Verses_English.csv'

# Load the CSV file into a Pandas DataFrame
aoy_df = pd.read_csv(aoy_file_path)
gita_df = pd.read_csv(gita_file_path)

#access chunks
# print(df['Chunk Content'][100])

#load saved embeddings on a CPU to make model predictions
aoy_embeddings = np.load('data/AOY/AOY_Embeddings_small.npy')
gita_embeddings = np.load('data/Gita/Bhagwad_Gita_Embeddings_small.npy')

#to download model files (downloaded folder - bge-large-en-v1.5)
# !git lfs install
# !git clone https://huggingface.co/BAAI/bge-large-en-v1.5

#move the model to a CPU
model = SentenceTransformer('BAAI/bge-small-en-v1.5').cpu()
top_k = 2
def retrieve_answers(question):
  query_instruction = "Represent this sentence for searching relevant passages: "
  query_embedding = model.encode([query_instruction + question])
  aoy_result = util.semantic_search(query_embedding, aoy_embeddings, top_k=top_k)
  gita_result = util.semantic_search(query_embedding, gita_embeddings, top_k=top_k)
    
  #Create Gita response
  gita_res = {}
  corpus_id = gita_result[0][0]['corpus_id']
  gita_res['Chapter'] = df['Chapter'][corpus_id]
  gita_res['Verse'] = df['Verse'][corpus_id]
  gita_res['Speaker'] = df['Speaker'][corpus_id]
  gita_res['Sanskrit'] = df['Sanskrit '][corpus_id]
  gita_res['English'] = df['Swami Sivananda'][corpus_id]
    
  #Create AOY response
  aoy_res = {}
  corpus_id = aoy_result[0][0]['corpus_id']
  aoy_res['Chunk Content'] = df['Chunk Content'][corpus_id]
  aoy_res['Chapter'] = df['Chapter'][corpus_id]
  
  return (gita_res, aoy_res)
