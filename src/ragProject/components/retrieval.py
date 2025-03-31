import faiss
from ..config.configuration import JD_Text,resume_text,system_prompt
from ..components.embeddings import resume_embedding
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_stores/faiss_job_descriptions.index"
jd_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_stores/job_descriptions.npy"


# Load FAISS index
index = faiss.read_index(index_path)

# Load stored job descriptions
job_descriptions = np.load(jd_file_path, allow_pickle=True)

def search_similar_jobs(resume_text, top_k=3):
    #print("Search similar jobs in Retreival starts")
    resume_embeddings = model.encode([resume_text], convert_to_numpy=True)
    distances, indices = index.search(resume_embeddings,top_k)


    """print("\nTop Matching Job Descriptions:")
    for i, idx in enumerate(indices[0]):
        print(f"\nRank {i+1}:")
        print(job_descriptions[idx])
        print(f"Similarity Score: {1 - distances[0][i]:.4f}")"""

    top_matches = [
        (job_descriptions[idx],1-distances[0][i]) # Convert L2 distance to similarity score
        for i, idx in enumerate(indices[0])
    ]
    return top_matches

# Example: Compare candidate resume with stored jobs
"""matching_jobs = search_similar_jobs(resume_text)

for rank, (job_desc, score) in enumerate(matching_jobs, start=1):
    print(f"\nRank {rank}:")
    print(job_desc)
    print(f"Similarity Score: {score:.4f}")"""



"""
# Create a FAISS index
dimension = job_embeddings.shape[1]  # Get embedding dimension
index = faiss.IndexFlatL2(dimension)
index.add(np.array(job_embeddings).astype('float32'))

# Search for top-K job descriptions
D, I = index.search(np.array([resume_embedding]).astype('float32'), K)

# Retrieve matching job descriptions
for idx in I[0]:
    print(f"Job Match: {job_descriptions[idx]}")
"""