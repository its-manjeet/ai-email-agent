import json
import yaml
import os
from google import genai  # CORRECT: Importing official Gemini library
from dotenv import load_dotenv
from typing import List


# Load environment variables
load_dotenv() 


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def generate_embeddings():
    config = load_config()

    # --- CORRECTION 1: Initialize the Gemini Client ---
    # client = genai.Client() automatically picks up the GEMINI_API_KEY environment variable.
    client = genai.Client() 

    # Load the vault content
    print("Loading vault content...")
    with open(config["vault_file"], "r", encoding="utf-8") as f:
        vault_text = f.read().split("\n")

    # Generate embeddings
    print("Generating embeddings... This may take a while.")
    embeddings = []
    
    # --- CORRECTION 2: Use the standard Gemini Embedding Model ---
    EMBEDDING_MODEL = "text-embedding-004" 
    
    for i, text in enumerate(vault_text):
        if text.strip():  # Skip empty lines
            try:
                # --- CRITICAL FIX 3a: Changed keyword 'content' to 'contents' ---
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL, contents=text  
                )
                # --- CRITICAL FIX 3b: FINAL CORRECTION: Access the vector via the correct plural attribute and index ---
                embeddings.append(response.embeddings[0]) 
            except Exception as e:
                # Catch any API errors but keep processing other emails
                print(f"Error processing line {i + 1}: {str(e)}")
                continue 
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} lines...")

    # Save embeddings
    print(f"Saving embeddings to {config['embeddings_file']}...")
    with open(config["embeddings_file"], "w") as f:
        json.dump(embeddings, f)

    print("Done! Embeddings have been generated and saved.")


if __name__ == "__main__":
    generate_embeddings()