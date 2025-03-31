# Implement the LLM integration here

#before going to cloud, below calls the local Llama2 from Olama
import requests
from ..config.configuration import system_prompt
from ..components.retrieval import search_similar_jobs

def resume_review(resume_text, JD_Text):
    #print("resume_review Starts")
    #print(resume_text)
    #print(JD_Text)
    # Retrieve Top 3 resumes m,atching candidate's resume
    matching_jobs = search_similar_jobs(resume_text)
    # convert the 3 JDs in text
    Top_3_JD = ""  # Initialize the string
    for rank, (job_desc, score) in enumerate(matching_jobs, start=1):
        Top_3_JD += f"\n?? Rank {rank}: {job_desc}\n"    

    prompt = (
        "You are an AI Resume Reviewer. Your task is to compare a candidate's resume with a job description and provide a structured evaluation.\n"
        "### **This is the Job Description (JD):** " + JD_Text + "\n"
        "### **This is the Candidate Resume:** " + resume_text + "\n"
        "### **Here are the top 3 industry job descriptions for more context:**.\n"
        + Top_3_JD + "\n"
        + system_prompt 
        )
    #print(prompt)
    # Call LLM to get the response
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    response_data = response.json()
    evaluation_summary = response_data.get("response", "No summary found.")
    #print(response.json()["response"])
    return evaluation_summary  
    """
    # Call AWS SageMaker endpoint
    import json
    import boto3
    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({"inputs": prompt})
    )
    response_data = json.loads(response["Body"].read().decode())
    evaluation_summary = response_data.get("generated_text", "No summary found.")

    return evaluation_summary"""

def followup(question, evaluation_summary, resume_text, jd_text):
    followup_prompt = (resume_text + "\n"
         + jd_text + "\n"
         + evaluation_summary + "\n"
         + question + "\n"
    )
    # Call LLM to get the response
    url = "http://localhost:11434/api/generate"
    followup_data = {
        "model": "llama2",
        "prompt": followup_prompt,
        "stream": False
    }
    response = requests.post(url, json=followup_data)
    response_data = response.json()
    followup_response = response_data.get("response", "No response found.")

    #print(response.json()["response"])
    return followup_response
