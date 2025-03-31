import pandas as pd
import boto3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from io import BytesIO

# S3 client initialization
s3_client = boto3.client('s3')
# Using Access Key and Secret Key in your Python script (not recommended for production)
s3_client = boto3.client(
    's3',
    aws_access_key_id='AKIAST2FMABP2ZIM4D7N',
    aws_secret_access_key='gc2bTAwg0mTTbW93Ihk5lJmCtYhPOUeRvXSqUbnC',
    region_name='us-east-1'  # Replace with your region
)


# Set your S3 bucket and folder paths
s3_bucket_name = 'vectorstorearchit'
s3_faiss_index_key = 'vector_store/faiss_store/faiss_job_descriptions.index'
s3_job_desc_key = 'vector_store/faiss_store/job_descriptions.npy'
s3_jd_file_key = 'vector_store/data/monster_com-job_sample.csv'  # S3 path to the CSV

# Download the JD file from S3 into memory
jd_file_object = s3_client.get_object(Bucket=s3_bucket_name, Key=s3_jd_file_key)

# Read the CSV file directly from S3 using pandas and BytesIO
jd_file_content = jd_file_object['Body'].read()
df_Job_descriptiosn = pd.read_csv(BytesIO(jd_file_content))

# Merging Job Title with Job Description
df_Job_descriptiosn["JDnTitle"] = df_Job_descriptiosn["job_title"] + ": " + df_Job_descriptiosn["job_description"]

# Create Vector Store for Industry standard Job Descriptions
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract the JD from the dataframe
job_descriptions = df_Job_descriptiosn["JDnTitle"].dropna().tolist()

# Generate embeddings for reference Job descriptions
jd_embeddings = model.encode(job_descriptions, convert_to_numpy=True)

# Create FAISS index (L2 normalization for efficient search)
dimension = jd_embeddings.shape[1] # Embedding size
index = faiss.IndexFlatL2(dimension)
index.add(jd_embeddings) # Add embeddings to the FAISS index

# --- Upload FAISS Index to S3 ---

# Save FAISS index

# --- Step 1: Write the FAISS Index to a Local File ---
faiss_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_stores/faiss_job_descriptions.index"
faiss.write_index(index, faiss_file_path)
print(f"FAISS index written to local file: {faiss_file_path}")
# --- Step 2: Upload the Local FAISS Index File to S3 ---
with open(faiss_file_path, 'rb') as f:
    s3_client.put_object(Body=f.read(), Bucket=s3_bucket_name, Key=s3_faiss_index_key)
    print(f"FAISS index uploaded to S3: {s3_faiss_index_key}")


# --- Step 1: Write the FAISS Index to a Local File ---
job_desc_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/vector_store/faiss_stores/job_descriptions.npy"
np.save(job_desc_file_path, job_descriptions)
# --- Step 2: Upload the Local FAISS Index File to S3 ---
with open(job_desc_file_path, 'rb') as f:
    s3_client.put_object(Body=f.read(), Bucket=s3_bucket_name, Key=s3_job_desc_key)
    print(f"FAISS index uploaded to S3: {s3_job_desc_key}")















"""
# Load the Industry Data into dataframe
JD_file_path = "C:/Archit/1Learning/IK/3 Capstone Project/Capstone/data/raw/monster_com-job_sample.csv"
df_Job_descriptiosn = pd.read_csv(JD_file_path)
# Merging Job Title with Job Description
df_Job_descriptiosn["JDnTitle"] = df_Job_descriptiosn["job_title"] + ": " + df_Job_descriptiosn["job_description"] 

# Create Vector Store for Industry standard Job Descriptions
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
"""
