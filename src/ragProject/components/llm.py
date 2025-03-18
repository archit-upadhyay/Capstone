# Implement the LLM integration here

#before going to cloud, below calls the local Llama2 from Olama
import requests
from config.configuration import JD_Text,resume_text,system_prompt

prompt = (
    "You are an AI Resume Reviewer. Your task is to compare a candidate's resume with a job description and provide a structured evaluation.\n"
    "### **Job Description (JD):** " + JD_Text + "\n"
    "### **Candidate Resume:** " + resume_text + 
    + system_prompt 
    )



url = "http://localhost:11434/api/generate"
data = {
    "model": "llama2",
    "prompt": prompt,
    "stream": False
}

response = requests.post(url, json=data)
print(response.json()["response"])