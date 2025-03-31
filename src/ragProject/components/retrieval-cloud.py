import faiss
from ..config.configuration import JD_Text,resume_text,system_prompt
from ..components.embeddings import resume_embedding
import numpy as np
from sentence_transformers import SentenceTransformer
import boto3

r = """John Doe
Email: johndoe@email.com

Work Experience:
- **Cloud Architect at AWS** (2018-Present)
- Designed scalable cloud solutions for enterprise clients.
- Led Kubernetes deployments for hybrid cloud systems.

Skills:
- **Cloud Computing, Kubernetes, Python, Terraform, AWS**

Education:
- **Master's in Computer Science, Stanford University**

Certifications:
- **AWS Certified Solutions Architect**
"""

# S3 client initialization
s3_client = boto3.client('s3')
# Using Access Key and Secret Key in your Python script (not recommended for production)
s3_client = boto3.client(
    's3',
    aws_access_key_id='AKIAST2FMABP2ZIM4D7N',
    aws_secret_access_key='gc2bTAwg0mTTbW93Ihk5lJmCtYhPOUeRvXSqUbnC',
    region_name='us-east-1'  
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# AWS S3 Configuration
s3_bucket_name = "vectorstorearchit"
s3_faiss_index_key = 'vector_store/faiss_store/faiss_job_descriptions.index'
s3_job_desc_key = 'vector_store/faiss_store/job_descriptions.npy'


# Local file paths
local_faiss_index_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_stores/faiss_job_descriptions.index"
local_job_desc_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_stores/job_descriptions.npy"
# --- Step 1: Download FAISS Index File ---
s3_client.download_file(s3_bucket_name, s3_faiss_index_key, local_faiss_index_path)
print(f"FAISS index downloaded to: {local_faiss_index_path}")

# --- Step 2: Download Job Descriptions File ---
s3_client.download_file(s3_bucket_name, s3_job_desc_key, local_job_desc_path)
print(f"Job descriptions downloaded to: {local_job_desc_path}")


#index_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_store/faiss_job_descriptions.index"
#jd_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_store/job_descriptions.npy"



# Load FAISS index
index = faiss.read_index(local_faiss_index_path)

# Load stored job descriptions
job_descriptions = np.load(local_job_desc_path, allow_pickle=True)

   
def search_similar_jobs(resume_text, top_k=3):
    print("resume_review Starts")
    resume_embedding = model.encode([resume_text], convert_to_numpy=True)
    distances, indices = index.search(resume_embedding,top_k)


    print("\nTop Matching Job Descriptions:")
    for i, idx in enumerate(indices[0]):
        print(f"\nRank {i+1}:")
        print(job_descriptions[idx])
        print(f"Similarity Score: {1 - distances[0][i]:.4f}")


    top_matches = [
        (job_descriptions[idx],1-distances[0][i]) # Convert L2 distance to similarity score
        for i, idx in enumerate(indices[0])
    ]
    return top_matches

"""    
# Create a FAISS index
dimension = job_embeddings.shape[1]  # Get embedding dimension
index = faiss.IndexFlatL2(dimension)
index.add(np.array(job_embeddings).astype('float32'))

# Search for top-K job descriptions
D, I = index.search(np.array([resume_embedding]).astype('float32'), K)

# Retrieve matching job descriptions
for idx in I[0]:
    print(f"Job Match: {job_descriptions[idx]}")"""


#search_similar_jobs(r,3)