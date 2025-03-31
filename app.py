from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from src.ragProject.components.llm import resume_review, followup  # Import RAG inference pipeline

app = FastAPI()

# Serve static files (optional: if you have styles or scripts in 'static/')
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Jinja2 templates from the 'templates' directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/review", response_class=HTMLResponse)
async def review_resume(request: Request, resume_file: UploadFile = File(...), jd_file: UploadFile = File(...)):
    print("Received request to /review")
    # Read uploaded files
    resume_text = await resume_file.read()
    jd_text = await jd_file.read()
    
    # Convert bytes to string
    resume_text = resume_text.decode("utf-8")
    jd_text = jd_text.decode("utf-8")

    
    # Call RAG inference pipeline
    evaluation_summary = resume_review(resume_text, jd_text)
    #print(evaluation_summary)
    return templates.TemplateResponse("results.html", {"request": request, "result": evaluation_summary})

@app.post("/followup", response_class=HTMLResponse)
async def followup_question(request: Request, 
                            question: str = Form(...),
                            evaluation_summary: str = Form(...),
                            resume_text: str = Form(...),
                            jd_text: str=Form(...)
                            ):
    # Mock response - Replace with LLM call
    #followup_response = f"AI Response to '{question}': This is a placeholder response."
    print(followup_question)
    followup_response = followup(question, evaluation_summary, resume_text, jd_text)
    print(followup_response)

    return templates.TemplateResponse("results.html", {"request": request, "result_followup": followup_response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
