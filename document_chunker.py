import re
import csv
import requests

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def parse_document(text):
    chapters = re.split(r'CHAPTER: \d+', text)[1:]  # Split text into chapters
    
    chapter_data = []
    for i, chapter in enumerate(chapters, start=1):
        chapter_lines = chapter.strip().split('\n')
        chapter_number = f"Chapter {i}"
        chapter_heading = chapter_lines[0].strip()
        chapter_content = '\n'.join(chapter_lines[1:]).strip()
        
        # Split chapter content into chunks
        chunks = split_into_chunks(chapter_content)
        
        for chunk in chunks:
            chapter_data.append({
                "Index number": len(chapter_data),
                "Chunk Content": chunk,
                "Chapter": f"{chapter_number} : {chapter_heading}"
            })
    
    return chapter_data

def split_into_chunks(text, max_chunk_size=750):
    paragraphs = re.split(r'\n\s*\n', text)  # Split text into paragraphs
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + '\n'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def write_to_csv(data, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["Index number", "Chunk Content", "Chapter"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)

# Example usage with file input
file_path = "AOY.txt"  # Replace with the actual file path
text_document = read_text_file(file_path)

output_data = parse_document(text_document)
output_csv_file = "AOY.csv"
write_to_csv(output_data, output_csv_file)
