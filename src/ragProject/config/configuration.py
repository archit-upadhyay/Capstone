import requests

resume_text = """John Doe
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
- **AWS Certified Solutions Architect** """


JD_Text = """Job Title: Technical Program Manager - Cloud

Responsibilities:
- Lead cloud transformation projects for enterprise customers.
- Manage cross-functional teams to deploy cloud-based architectures.

Required Skills:
- **GCP, AI/ML, Cloud Architecture, Kubernetes**
- Experience with **multi-cloud environments**.

Required Experience:
- **4+ years in cloud program management**.

Education:
- **Bachelor’s degree or higher in Computer Science**.

Preferred Certifications:
- **GCP Professional Cloud Architect**"""


system_prompt =     """### **Evaluation Criteria:**\n"
    "1. **Skills Match:** Identify overlapping and missing skills.\n"
    "2. **Experience Relevance:** Compare the candidate's work experience with job requirements.\n"
    "3. **Education Alignment:** Verify if the candidate meets the education requirement.\n"
    "4. **Certifications & Extras:** Identify relevant or missing certifications.\n"
    "5. **Job Fit Score (0-100):** Assign a match score based on the alignment.\n"
    "6. **Improvement Suggestions:** Provide recommendations for better alignment.\n"
    "Now, analyze the given resume against the job description and provide a structured response based on each evaluation criteria."
)"""
