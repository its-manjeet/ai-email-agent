import json
import yaml
import os
from google import genai # <-- CORRECTED: Switched to Gemini SDK
from dotenv import load_dotenv
import torch
from torch import Tensor
import numpy as np
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# --- CONFIGURATION CONSTANTS (Gemini Model Names) ---
EMBEDDING_MODEL_GEMINI = "text-embedding-004"
CHAT_MODEL_GEMINI = "gemini-2.5-flash"


def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    if not isinstance(b, Tensor):
        b = torch.tensor(b)
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


class LocalRAG:
    def __init__(self):
        self.config = load_config()
        
        # --- CRITICAL FIX 1: Initialize Gemini client, which auto-detects GEMINI_API_KEY ---
        self.client = genai.Client()
        
        # Set the model names in config to ensure the API calls use the correct names
        self.config.setdefault("gemini", {})
        self.config["gemini"]["embedding_model"] = EMBEDDING_MODEL_GEMINI
        self.config["gemini"]["chat_model"] = CHAT_MODEL_GEMINI
        
        self.load_vault()

    def load_vault(self):
        # Check if vault file exists
        if not os.path.exists(self.config["vault_file"]):
            raise FileNotFoundError(
                f"Vault file {self.config['vault_file']} not found. Please run collect_emails.py first."
            )

        # Load the text content
        with open(self.config["vault_file"], "r", encoding="utf-8") as f:
            self.vault_text = f.read().split("\n")

        # Check if embeddings file exists, if not, generate embeddings
        if not os.path.exists(self.config["embeddings_file"]):
            print("Embeddings file not found. Generating embeddings...")
            self.generate_embeddings()

        # Load the embeddings
        with open(self.config["embeddings_file"], "r") as f:
            self.embeddings = json.load(f)

        # Convert embeddings to tensor
        self.embeddings_tensor = torch.tensor(self.embeddings)

    def generate_embeddings(self):
        """Generate embeddings for the vault content"""
        print("Generating embeddings... This may take a while.")
        embeddings = []
        for i, text in enumerate(self.vault_text):
            if text.strip():  # Skip empty lines
                # --- CRITICAL FIX 2a: Use genai.Client and correct keywords ---
                response = self.client.models.embed_content(
                    model=EMBEDDING_MODEL_GEMINI, contents=text
                )
                # --- CRITICAL FIX 2b: Access the embedding vector array via standard SDK attribute ---
                embeddings.append(response.embeddings[0]) 
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} lines...")

        # Save embeddings
        with open(self.config["embeddings_file"], "w") as f:
            json.dump(embeddings, f)
        print("Embeddings generated and saved successfully!")

    def get_embedding(self, text: str) -> List[float]:
        """Get query embedding using Gemini's embedding model"""
        # --- CRITICAL FIX 3: Use genai.Client and correct return structure ---
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL_GEMINI, contents=text
        )
        return response.embeddings[0]

    def get_relevant_context(self, query: str) -> str:
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Calculate similarities
        similarities = [
            cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in self.embeddings
        ]

        # Get top-k most similar indices
        top_k = self.config["top_k"]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Construct context from top-k most similar documents
        context = "\n".join([self.vault_text[i] for i in top_indices])
        return context

    def query(self, user_input: str) -> str:
        # Get relevant context
        context = self.get_relevant_context(user_input)

        # Construct the messages for the chat model
        messages = [
            {"role": "user", "content": self.config["system_message"]},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_input}",
            },
        ]

        # Get response from the chat model
        # --- CRITICAL FIX 4: Use genai.Client.models.generate_content ---
        response = self.client.models.generate_content(
            model=CHAT_MODEL_GEMINI,
            contents=messages, # Gemini standard uses 'contents' keyword
            config={
                "temperature": self.config["openai"]["temperature"],
                "max_output_tokens": self.config["openai"]["max_tokens"],
            }
        )

        return response.text # Gemini response text is accessed via .text


def main():
    try:
        rag = LocalRAG()
        print("Vault loaded successfully. RAG system ready!")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run the data collection script first: 'python collect_emails.py'")
        return
        
    print("\nWelcome to LocalRAG! Type 'quit' to exit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == "quit":
            break

        try:
            response = rag.query(user_input)
            print("\nResponse:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
