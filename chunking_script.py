import os
import numpy as np
import logging
from datetime import datetime
import pandas as pd
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from bson import ObjectId
from openai import OpenAI
import json
import yaml
import seaborn as sns
import matplotlib.pyplot as plt


# Set up logging to a file
logging.basicConfig(level=logging.INFO, filename=config.get('log_file', 'app.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s')



def read_file_id(file_path):
    """Reads the file_id from a file."""
    try:
        with open(file_path, 'r') as file:
            file_id = file.readline().strip()
        if not file_id:
            raise ValueError("The file_id file is empty.")
        logging.info(f"Read file_id: {file_id}")
        print(f"Read file_id: {file_id}")
        return file_id
    except Exception as e:
        logging.error(f"Failed to read file_id: {e}")
        raise

file_id = read_file_id(config.get('file_id_path', 'current_file_id.txt'))

def fetch_cleaned_transcription(db, file_id):
    """Fetches the raw transcription document from MongoDB."""
    try:
        transcription_doc = db.fs.removed_ad_lingo_transcription.find_one({"original_file_id": ObjectId(file_id)})
        if not transcription_doc:
            raise FileNotFoundError(f"No transcription found for original file_id {file_id}")
        logging.info(f"Fetched document: {transcription_doc}")
        raw_data = transcription_doc.get("raw_transcription", None)
        if raw_data is None:
            raise ValueError("The 'raw_transcription' field is missing in the fetched document.")
        logging.info(f"Raw data length: {len(raw_data)}")
        logging.info(f"Raw data snippet: {raw_data[:500]}")
        if not raw_data.strip():
            raise ValueError("Fetched data is empty or missing.")
        txt = raw_data
        logging.info(f"Successfully fetched raw transcription for original file_id: {file_id}")
        return txt
    except Exception as e:
        logging.error(f"Failed to fetch raw transcription from MongoDB: {e}")
        raise

txt = fetch_cleaned_transcription(db, file_id)

# Split the transcript into sentences
sentences = txt.split('.')
sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]

# Create a list of dictionaries for each sentence
sentence_data = []
for i, sentence in enumerate(sentences):
    sentence_data.append({
        'sentence_num': i,
        'text': sentence,
    })

response = client.embeddings.create(
    input=sentences,
    model="text-embedding-3-small"
)

embeddings = [data.embedding for data in response.data]

def cosine_similarity_next_five(embeddings):
    similarities = []
    for i in range(len(embeddings) - 2):  # Subtract 2 to ensure we have 3 sentences
        current_similarities = []
        current_embeddings = embeddings[i:i+3]  # Get 3 consecutive embeddings
        
        # Compare with the next up to 5 sentences
        for j in range(1, 6):
            if i + 2 + j < len(embeddings):
                next_embedding = embeddings[i + 2 + j]
                # Calculate average similarity with the 3 current sentences
                sims = [1 - cosine(curr_emb, next_embedding) for curr_emb in current_embeddings]
                avg_sim = sum(sims) / len(sims)
                current_similarities.append(avg_sim)
        
        similarities.append(current_similarities)
    
    return similarities

# Assuming you have your embeddings stored in a list called 'embeddings'
similarities = cosine_similarity_next_five(embeddings)

def linear_weights(window_size, weight_current):
    """Generate linear weights for the window."""
    return np.linspace(weight_current, 1, window_size)

def exponential_weights(window_size, weight_current):
    """Generate exponential weights for the window."""
    return np.exp(np.linspace(0, -weight_current, window_size))

# Configuration Section
WINDOW_SIZE = 5
WEIGHT_CURRENT = 1.2
STD_FACTOR = 1.6
SMOOTHING_WINDOW = 1
USE_WEIGHT_FUNC = True  # Set this to False to disable weighting
WEIGHT_FUNC = (lambda x: exponential_weights(x, WEIGHT_CURRENT)) if USE_WEIGHT_FUNC else lambda x: np.ones(x)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def trailing_average_similarity(embeddings):
    similarities = cosine_similarity_next_five(embeddings)
    trailing_averages = []
    chunk_points = []

    for i in range(len(similarities)):
        current_similarity = similarities[i]

        if not current_similarity:
            continue

        # Apply dynamic weighting if enabled
        if USE_WEIGHT_FUNC:
            weights = WEIGHT_FUNC(len(current_similarity))
            if np.sum(weights) == 0:
                weights = np.ones(len(current_similarity))
            weighted_similarity = np.average(current_similarity, weights=weights)
        else:
            weighted_similarity = np.mean(current_similarity)

        # Calculate the trailing average
        if i > 0:
            flattened_similarities = [item for sublist in similarities[max(0, i-WINDOW_SIZE):i] for item in sublist]
            trailing_avg = np.mean(flattened_similarities)
            trailing_std = np.std(flattened_similarities)
        else:
            trailing_avg = weighted_similarity
            trailing_std = np.std(current_similarity)

        # Adjust the trailing average by giving the current sentence a higher weight
        weighted_avg = (trailing_avg * (WEIGHT_CURRENT - 1) + weighted_similarity) / WEIGHT_CURRENT
        trailing_averages.append(weighted_avg)

        # Dynamic divergence detection based on standard deviation
        if i >= WINDOW_SIZE and i + WINDOW_SIZE < len(similarities):
            next_avg = np.mean([item for sublist in similarities[i:i+WINDOW_SIZE] for item in sublist])
            if np.abs(next_avg - weighted_avg) > STD_FACTOR * trailing_std:
                chunk_points.append(i + 2)  # Add 2 to account for the 3-sentence window

    # Apply smoothing to trailing averages
    smoothed_trailing_averages = moving_average(trailing_averages, SMOOTHING_WINDOW)

    # Adjust chunk points to match the length of smoothed_trailing_averages
    adjusted_chunk_points = [cp for cp in chunk_points if cp < len(smoothed_trailing_averages)]

    return adjusted_chunk_points, smoothed_trailing_averages, similarities[:len(smoothed_trailing_averages)]

# Use the method with embeddings
chunk_points, trailing_averages, similarities = trailing_average_similarity(embeddings)

# Group sentences into chunks
chunks = []
current_chunk = []
chunk_index = 0
for i, sentence in enumerate(sentences):
    current_chunk.append(sentence)
    if chunk_index < len(chunk_points) and i == chunk_points[chunk_index]:
        chunks.append(current_chunk)
        current_chunk = []
        chunk_index += 1

# Append any remaining sentences in the last chunk
if current_chunk:
    chunks.append(current_chunk)

# Save to JSON
output_data = {
    "chunks": [
        {
            "chunk_num": i,
            "sentences": chunk
        } for i, chunk in enumerate(chunks)
    ]
}

output_file = 'chunked_transcript.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)
print(f"Chunked sentences have been saved to {output_file}.")

# Visualization
sentence_indices = list(range(len(embeddings)))
flattened_similarities = [item for sublist in similarities for item in sublist]

min_length = min(len(sentence_indices), len(flattened_similarities), len(trailing_averages))
sentence_indices = sentence_indices[:min_length]
flattened_similarities = flattened_similarities[:min_length]
trailing_averages = trailing_averages[:min_length]

plt.figure(figsize=(14, 8))
plt.plot(sentence_indices, flattened_similarities, label='Cosine Similarities', color='blue', alpha=0.7, linewidth=1.5)
plt.plot(sentence_indices, trailing_averages, label='Trailing Averages', color='green', alpha=0.7, linewidth=1.5)

for chunk_point in chunk_points:
    plt.axvline(x=chunk_point, color='red', linestyle='--', alpha=0.3, label='Chunk Point' if chunk_point == chunk_points[0] else "")

plt.xlabel('Sentence Index')
plt.ylabel('Value')
plt.title('Cosine Similarities, Trailing Averages, and Chunk Points')
plt.legend()
plt.show()
