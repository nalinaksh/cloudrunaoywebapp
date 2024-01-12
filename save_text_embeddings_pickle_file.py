import pandas as pd
import numpy as np
import pickle

def create_and_save_dataframe(result_chunks, corpus_embeddings, pickle_filename="dataframe_with_embeddings.pkl"):
    # Extract text and create a DataFrame
    df = pd.DataFrame({'text': [chunk for chunk in result_chunks]})
    
    # Add the 'embedding' column as strings without square brackets
    df['embedding'] = [embedding for embedding in corpus_embeddings]

    # Save the DataFrame to a pickle file
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(df, pickle_file)

    return df

def load_dataframe_from_pickle(pickle_filename="dataframe_with_embeddings.pkl"):
    # Load DataFrame from pickle file
    with open(pickle_filename, 'rb') as pickle_file:
        df = pickle.load(pickle_file)

    return df

# Example usage
# Create DataFrame and save to CSV and pickle
df = create_and_save_dataframe(result_chunks, corpus_embedding, pickle_filename="dataframe_with_embeddings.pkl")
print("DataFrame with embeddings saved to CSV and pickle files.")

# Load DataFrame from pickle file
loaded_df = load_dataframe_from_pickle("dataframe_with_embeddings.pkl")
print("\nLoaded DataFrame from pickle file:")
# print(loaded_df)

#access text and embeddings
print(loaded_df["text"][10])
print(loaded_df["embedding"][10])
