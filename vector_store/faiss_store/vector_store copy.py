import pandas as pd

# Load the Industry Data into dataframe
JD_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/data/raw/monster_com-job_sample.csv"
df_Job_descriptiosn = pd.read_csv(JD_file_path)
# Merging Job Title with Job Description
df_Job_descriptiosn["JDnTitle"] = df_Job_descriptiosn["job_title"] + ": " + df_Job_descriptiosn["job_description"] 

# Create Vector Store for Industry standard Job Descriptions
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract the JD from the dataframe
job_descriptions = df_Job_descriptiosn["JDnTitle"].dropna().tolist()

# Generate embeddings for reference Job descriptions
jd_embeddings = model.encode(job_descriptions, convert_to_numpy=True)

# Create FAISS index (L2 normalization for efficient search)
dimension = jd_embeddings.shape[1] # Embedding size
index = faiss.IndexFlatL2(dimension)
index.add(jd_embeddings) # Add embeddings to the FAISS index

# Save FAISS index
faiss_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_store/faiss_job_descriptions.index"
faiss.write_index(index, faiss_file_path)

# Save job descriptions separately for retrieval
job_desc_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_store/job_descriptions.npy"
np.save(job_desc_file_path, job_descriptions)
