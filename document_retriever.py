from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from openai import OpenAI
import pandas as pd
import numpy as np
import os

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

# Semantic search
def retrieve_results(question, corpus_embeddings, top_k):
  query_instruction = "Represent this sentence for searching relevant passages: "
  query_embedding = model.encode([query_instruction + question])
  result = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
  return result

# For authentic responses 
def retrieve_answers(question):
  aoy_result = retrieve_results(question, aoy_embeddings, top_k=2)
  gita_result = retrieve_results(question, gita_embeddings, top_k=2)
    
  #For authentic Gita response
  gita_res = {}
  corpus_id = gita_result[0][0]['corpus_id']
  gita_res['Score'] = gita_result[0][0]['score']
  gita_res['Chapter'] = gita_df['Chapter'][corpus_id]
  gita_res['Verse'] = gita_df['Verse'][corpus_id]
  gita_res['Speaker'] = gita_df['Speaker'][corpus_id]
  gita_res['Sanskrit'] = gita_df['Sanskrit '][corpus_id]
  gita_res['English'] = gita_df['Swami Sivananda'][corpus_id]
    
  #For authentic AOY response
  aoy_res = {}
  corpus_id = aoy_result[0][0]['corpus_id']
  aoy_res['Score'] = aoy_result[0][0]['score']
  aoy_res['Chunk Content'] = aoy_df['Chunk Content'][corpus_id]
  aoy_res['Chapter'] = aoy_df['Chapter'][corpus_id]
  
  return (gita_res, aoy_res)

# For Gita consultation (using Gen AI)
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_prompt(query, gita_context):
  prompt = f"""
  Query:
  {query}

  Gita context:
  {gita_context}
  """
  return prompt

def fetch_counsel(question):
  top_k = 10
  gita_result = retrieve_results(question, gita_embeddings, top_k=top_k)
  #Create Gita response
  gita_context = ""
  for i in range(top_k):
    corpus_id = gita_result[0][i]['corpus_id']
    gita_context += gita_df['Swami Sivananda'][corpus_id]
    gita_context += "\n"

  prompt = get_prompt(question, gita_context)
  
  messages = [{"role": "system", "content": "You are a helpful assistance. \
  The context will provide you a user query followed by some passages from the Indian scripture of Bhagvad Gita, \
  which contains dialogue between Lord Krishna and devotee Arjuna. You should try to find if you can generate an \
  answer to a user query in the context of the Gita dialogue. If the answer is not in the given context, tell the \
  user so. Do not try to make up the answer or consult any other source."}]
  
  messages.append({"role": "user", "content": prompt})

  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=256,
    stream=False)

  return completion.choices[0].message.content

