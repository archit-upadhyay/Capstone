from sentence_transformers import SentenceTransformer
from config.configuration import JD_Text,resume_text,system_prompt
import pandas as pd

# Load a pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for the user's resume
resume_embedding = model.encode(resume_text, normalize_embeddings=True)

# print(resume_embedding.shape)  # (384,) for MiniLM