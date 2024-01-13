from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'AOY_text_chunks.csv'

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

#access chunks
print(df['Chunk Content'][100])

#to download model files (downloaded folder - bge-large-en-v1.5)
# !git lfs install
# !git clone https://huggingface.co/BAAI/bge-large-en-v1.5

#move the model to a CPU
model = SentenceTransformer('bge-large-en-v1.5').cpu()

#generate embeddings
corpus_embedding = model.encode(df['Chunk Content'], convert_to_tensor=False)

#save embeddings in a file as numpy array
np.save('AOY_Embeddings.npy', corpus_embedding)

#use saved embeddings on a CPU to make model predictions
all_embeddings = np.load('AOY_Embeddings.npy')

question = "What is Kriya Yoga?"
query_instruction = "Represent this sentence for searching relevant passages: "
query_embedding = model.encode([query_instruction + question])

print(util.semantic_search(query_embedding, all_embeddings, top_k=3))
