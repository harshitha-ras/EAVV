from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI(title="Weather Condition Object Detection API")

# Load your trained model
model = YOLO("yolo_output/yolov8s_weather_refined/weights/best.pt")

# Configure CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint to detect objects in an image with weather condition awareness
    """
    try:
        # Read image
        contents = await file.read()
        image = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Perform detection
        results = model(image)
        
        # Process results
        detections = []
        for i, det in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(float, det.xyxy[0])
            confidence = float(det.conf[0])
            class_id = int(det.cls[0])
            class_name = model.names[class_id]
            
            detections.append({
                "id": i,
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })
            
        return {"detections": detections}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "YOLOv8s Weather Refined"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
