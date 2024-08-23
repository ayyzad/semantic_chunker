import os
import numpy as np
import logging
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from openai import OpenAI
import json
import yaml
import matplotlib.pyplot as plt

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load environment variables from .env file
load_dotenv(config.get('env_file_path', '.env'))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, filename=config.get('log_file', 'app.log'),
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# MongoDB connection
def connect_to_mongo():
    try:
        mongo_client = MongoClient(os.getenv("MONGODB_URI"))
        db = mongo_client[config.get('mongodb_database', 'qpn_content_management')]
        logging.info("Connected to MongoDB")
        return db
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise

db = connect_to_mongo()

def read_file_id(file_path):
    """Reads the file_id from a file."""
    try:
        with open(file_path, 'r') as file:
            file_id = file.readline().strip()
        if not file_id:
            raise ValueError("The file_id file is empty.")
        logging.info(f"Read file_id: {file_id}")
        return file_id
    except Exception as e:
        logging.error(f"Failed to read file_id: {e}")
        raise

file_id = read_file_id(config.get('file_id_path', 'current_file_id.txt'))

def fetch_cleaned_transcription(db, file_id):
    """Fetches the transcription document from MongoDB."""
    try:
        transcription_doc = db.fs.removed_ad_lingo_transcription.find_one({"original_file_id": ObjectId(file_id)})
        if not transcription_doc:
            raise FileNotFoundError(f"No transcription found for original file_id {file_id}")
        segmented_sentences = transcription_doc.get("segmented_sentences")
        if not segmented_sentences:
            raise ValueError("Fetched data is empty or missing 'segmented_sentences'.")
        logging.info(f"Successfully fetched segmented sentences for original file_id: {file_id}")
        return segmented_sentences
    except Exception as e:
        logging.error(f"Failed to fetch segmented sentences from MongoDB: {e}")
        raise

segmented_sentences = fetch_cleaned_transcription(db, file_id)

def filter_sentences(segmented_sentences):
    """Filters sentences based on ad criteria."""
    return [
        sentence['text']
        for sentence in segmented_sentences
        if not (sentence['ad'] == 'yes' and sentence['confidence is ad'] > 0.84)
    ]

sentences = filter_sentences(segmented_sentences)

def generate_embeddings(sentences):
    """Generates embeddings for sentences."""
    try:
        response = client.embeddings.create(
            input=sentences,
            model="text-embedding-3-small"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        raise

embeddings = generate_embeddings(sentences)

def cosine_similarity_next_five(embeddings):
    """Calculates cosine similarities for the next five sentences."""
    similarities = []
    num_embeddings = len(embeddings)
    
    for i in range(num_embeddings - 2):
        current_embeddings = embeddings[i:i+3]
        current_similarities = [
            np.mean([1 - cosine(emb, embeddings[i + 2 + j]) for emb in current_embeddings])
            for j in range(1, 6) if i + 2 + j < num_embeddings
        ]
        similarities.append(current_similarities)
    
    return similarities

def calculate_weights(window_size, weight_current, weight_func_type):
    """Calculates weights based on the type of function (linear or exponential)."""
    if weight_func_type == 'linear':
        return np.linspace(weight_current, 1, window_size)
    elif weight_func_type == 'exponential':
        return np.exp(np.linspace(0, -weight_current, window_size))
    else:
        raise ValueError(f"Unsupported weight function type: {weight_func_type}")

def moving_average(data, window_size):
    """Calculates the moving average over a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def trailing_average_similarity(embeddings, window_size, smoothing_window, weight_current, std_factor, weight_func_type):
    """Calculates trailing averages and identifies chunk points."""
    similarities = cosine_similarity_next_five(embeddings)
    trailing_averages = []
    chunk_points = []
    weight_func = lambda x: calculate_weights(x, weight_current, weight_func_type)

    for i in range(len(similarities)):
        current_similarity = similarities[i]
        if not current_similarity:
            continue

        weights = weight_func(len(current_similarity))
        weighted_similarity = np.average(current_similarity, weights=weights)

        if i > 0:
            flattened_similarities = [item for sublist in similarities[max(0, i - window_size):i] for item in sublist]
            trailing_avg = np.mean(flattened_similarities)
            trailing_std = np.std(flattened_similarities)
        else:
            trailing_avg = weighted_similarity
            trailing_std = np.std(current_similarity)

        weighted_avg = (trailing_avg * (weight_current - 1) + weighted_similarity) / weight_current
        trailing_averages.append(weighted_avg)

        if i >= window_size and i + window_size < len(similarities):
            next_avg = np.mean([item for sublist in similarities[i:i + window_size] for item in sublist])
            if np.abs(next_avg - weighted_avg) > std_factor * trailing_std:
                chunk_points.append(i + 2)

    smoothed_trailing_averages = moving_average(trailing_averages, smoothing_window)
    adjusted_chunk_points = [cp for cp in chunk_points if cp < len(smoothed_trailing_averages)]

    return adjusted_chunk_points, smoothed_trailing_averages, similarities[:len(smoothed_trailing_averages)]

# Configuration Parameters
WINDOW_SIZE = 5
SMOOTHING_WINDOW = 1
WEIGHT_CURRENT = 0.75
WEIGHT_FUNC_TYPE = 'exponential'  # Change to 'linear' for linear weights
STD_FACTOR = 1.6

chunk_points, trailing_averages, similarities = trailing_average_similarity(
    embeddings, 
    window_size=WINDOW_SIZE, 
    smoothing_window=SMOOTHING_WINDOW, 
    weight_current=WEIGHT_CURRENT, 
    std_factor=STD_FACTOR,
    weight_func_type=WEIGHT_FUNC_TYPE
)

def group_sentences_into_chunks(sentences, chunk_points):
    """Groups sentences into chunks based on identified chunk points."""
    chunks = []
    current_chunk = []
    chunk_index = 0
    
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        if chunk_index < len(chunk_points) and i == chunk_points[chunk_index]:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_index += 1
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

chunks = group_sentences_into_chunks(sentences, chunk_points)

def save_chunks_to_json(chunks, output_file='chunked_transcript_WCpoint75.json'):
    """Saves the chunks into a JSON file."""
    output_data = {
        "chunks": [{"chunk_num": i, "sentences": chunk} for i, chunk in enumerate(chunks)]
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    logging.info(f"Chunked sentences have been saved to {output_file}.")

save_chunks_to_json(chunks)

def plot_similarities_and_trailing_averages(sentence_indices, flattened_similarities, trailing_averages, chunk_points):
    """Visualizes cosine similarities, trailing averages, and chunk points."""
    plt.figure(figsize=(14, 8))
    plt.plot(sentence_indices, flattened_similarities, label='Cosine Similarities', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(sentence_indices, trailing_averages, label='Trailing Averages', color='green', alpha=0.7, linewidth=1.5)

    mean_similarity = np.mean(flattened_similarities)
    plt.axhline(y=mean_similarity, color='purple', linestyle='-', linewidth=1.5, label='Mean Similarity')

    std_dev = np.std(flattened_similarities)
    for i in range(1, 4):
        plt.fill_between(sentence_indices, mean_similarity + i * std_dev, mean_similarity - i * std_dev, color='gray', alpha=0.1 * i, label=f'{i} Std Dev' if i == 1 else "")

    for chunk_point in chunk_points:
        plt.axvline(x=chunk_point, color='red', linestyle='--', alpha=0.3, label='Chunk Point' if chunk_point == chunk_points[0] else "")

    plt.xlabel('Sentence Index')
    plt.ylabel('Value')
    plt.title('Cosine Similarities, Trailing Averages, and Chunk Points')
    plt.legend()
    plt.show()

sentence_indices = list(range(len(embeddings)))
flattened_similarities = [item for sublist in similarities for item in sublist]

min_length = min(len(sentence_indices), len(flattened_similarities), len(trailing_averages))
sentence_indices = sentence_indices[:min_length]
flattened_similarities = flattened_similarities[:min_length]
trailing_averages = trailing_averages[:min_length]

plot_similarities_and_trailing_averages(sentence_indices, flattened_similarities, trailing_averages, chunk_points)
