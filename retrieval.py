from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

import time
import langchain
import subprocess
import os

#Set to True for debugging
langchain.verbose = False

###START To use the saved embeddings on a CPU
#Extract saved embeddings here
cmdline = ['/bin/tar','xvzf','./data/HuggingFaceEmbeddings.tar.gz']
subprocess.call(cmdline)
time.sleep(2)

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

persist_dir = "HuggingFaceEmbeddings"
vectorstore = None
vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_dir)
###END

def get_relevant_doc(query):
  docs = vectorstore.similarity_search(query)
  if len(docs) == 0:
    return "No doc found"
  return docs[0].page_content
