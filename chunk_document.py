import requests
import re

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def create_chunks(text, max_chunk_size=750):
    # Split the text into paragraphs based on a combination of newline characters and spaces
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If a single paragraph exceeds the max size, include it in the current chunk without checking the limit
        if len(paragraph) > max_chunk_size:
            chunks.append(paragraph.strip() + "\n\n")
            current_chunk = ""
        else:
            # Check if adding the current paragraph to the current chunk exceeds the max size
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                # Add the paragraph to the current chunk
                current_chunk += paragraph + "\n\n"
            else:
                # If adding the current paragraph exceeds the max size, start a new chunk with the current paragraph
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_chunks_from_url(url, max_chunk_size=750):
    text_content = fetch_text_from_url(url)

    if text_content is not None:
        return create_chunks(text_content, max_chunk_size)
    else:
        return None

# Example usage
url = "https://www.gutenberg.org/cache/epub/7452/pg7452.txt"
result_chunks = create_chunks_from_url(url)

if result_chunks is not None:
    for i, chunk in enumerate(result_chunks[100:110], 1):
        print(f"Chunk {i}:\n{chunk}\n")
