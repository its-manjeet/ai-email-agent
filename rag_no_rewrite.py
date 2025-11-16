import torch
import os
import json
from google import genai  # <-- CORRECTED: Importing the official Gemini library
import argparse
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any

# ANSI escape codes for colors
PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# Load environment variables
load_dotenv()

# --- CONFIGURATION CONSTANTS (Gemini Model Names) ---
EMBEDDING_MODEL_GEMINI = "text-embedding-004"
CHAT_MODEL_GEMINI = "gemini-2.5-flash"


# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return infile.read()


# Function to get relevant context from the vault based on user input
def get_relevant_context(
    rewritten_input, vault_embeddings, vault_content, client, top_k=3
):
    if vault_embeddings.nelement() == 0: 
        return []

    # Get embedding for the input using Gemini
    try:
        # --- CRITICAL FIX 1: Use genai.Client and correct keywords ---
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_GEMINI, contents=rewritten_input
        )
        # --- CRITICAL FIX 2: Access the embedding vector via correct plural attribute and index ---
        input_embedding = response.embeddings[0]
    except Exception as e:
        print(f"Error retrieving embedding for context: {str(e)}")
        return []


    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(
        torch.tensor(input_embedding).unsqueeze(0), vault_embeddings
    )

    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))

    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to interact with the Gemini model
def chat_with_gpt(
    user_input,
    system_message,
    vault_embeddings,
    vault_content,
    model,
    conversation_history,
    client,
):
    # This calls the globally defined vault_embeddings_tensor in the main function's scope.
    # This is a bit risky in Python but matches your friend's original structure.
    global vault_embeddings_tensor 
    
    relevant_context = get_relevant_context(
        user_input, vault_embeddings_tensor, vault_content, client, top_k=3
    )

    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = f"Context:\n{context_str}\n\nQuestion: {user_input}"

    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})

    # Create a message history including the system message and the conversation history
    # Note: Gemini often prefers a single 'contents' structure rather than a list of {role: content}
    full_prompt = [{"role": "system", "content": system_message}, *conversation_history]


    # Send the completion request to the Gemini model
