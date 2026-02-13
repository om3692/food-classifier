from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model_utils import classifier # Import our modular AI logic

# Initialize the FastAPI app
app = FastAPI(title="Food Freshness Classifier")

# 1. Mount the static directory (This fixes the 'Not Found' error for the logo)
# This tells FastAPI: "If anyone asks for /static, look in the static folder"
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory for serving HTML
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the frontend HTML page when you visit http://127.0.0.1:8000
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_freshness(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image, passes it to the model, 
    and returns the JSON prediction.
    """
    # Read the file bytes
    image_data = await file.read()
    
    # Get prediction from our modular utility
    result = classifier.predict(image_data)
    
    return result

# --- CRITICAL SECTION ---
# This block checks if this file is being run directly by Python.
# If it is, it starts the Uvicorn server.
if __name__ == "__main__":
    import uvicorn
    # This will print "Uvicorn running on http://127.0.0.1:8000" in your terminal
    uvicorn.run(app, host="127.0.0.1", port=8000)