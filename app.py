from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

"""app = FastAPI()

# Load Jinja2 templates from the "templates" folder
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
"""




from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.ragProject.pipeline.inference_pipeline import resume_reviewer  # Import your RAG inference pipeline

app = FastAPI()

# Serve static files (optional: if you have styles or scripts in 'static/')
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Jinja2 templates from the 'templates' directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/review", response_class=HTMLResponse)
async def review_resume(request: Request, resume_text: str = Form(...)):
    # Call the RAG pipeline for inference
    result = resume_reviewer(resume_text)
    
    return templates.TemplateResponse("results.html", {"request": request, "result": result})

# Optional: API Endpoint for JSON response
@app.post("/api/review")
async def api_review(resume_text: str = Form(...)):
    result = resume_reviewer(resume_text)
    return {"analysis": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

