from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import io
import base64
import cv2
import uvicorn
import os
from backend.services.dicom_processor import DicomProcessor
from backend.services.inference import InferenceService

app = FastAPI(title="ICU Tube Displacement API", version="1.0.0")

# Mount frontend
if not os.path.exists("frontend"):
    os.makedirs("frontend")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

inference_service = InferenceService()

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.dcm'):
        raise HTTPException(status_code=400, detail="Only DICOM (.dcm) files are supported")
        
    try:
        file_bytes = await file.read()
        image_rgb = DicomProcessor.process_dicom(file_bytes)
        results = inference_service.predict(image_rgb)
        annotated_image = results.pop("annotated_image")
        
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', annotated_bgr)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        results["image_base64"] = image_base64
        
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
