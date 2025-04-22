import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  
  const API_URL = 'http://YOUR_VM_IP:8000'; // Replace with your VM's IP

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setDetections([]);
    }
  };

  const drawDetections = (detections) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = preview;
    
    img.onload = () => {
      // Set canvas dimensions to match image
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Draw the image first
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // Draw each detection
      detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Draw bounding box
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw label background
        ctx.fillStyle = '#00FF00';
        const label = `${det.class} ${(det.confidence * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
        
        // Draw label text
        ctx.fillStyle = '#000000';
        ctx.font = '18px Arial';
        ctx.fillText(label, x1 + 5, y1 - 7);
      });
    };
  };

  const detectObjects = async () => {
    if (!image) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', image);
      
      const response = await fetch(`${API_URL}/detect/`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Detection failed');
      }
      
      const data = await response.json();
      setDetections(data.detections);
      drawDetections(data.detections);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Weather Condition Object Detection</h1>
        <p>Upload an image to detect objects in various weather conditions</p>
      </header>
      
      <div className="upload-section">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageChange} 
          className="file-input"
        />
        <button 
          onClick={detectObjects} 
          disabled={!image || isLoading}
          className="detect-button"
        >
          {isLoading ? 'Processing...' : 'Detect Objects'}
        </button>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="results-container">
        {preview && (
          <div className="image-container">
            <canvas ref={canvasRef} className="detection-canvas" />
            {detections.length > 0 && (
              <div className="detections-list">
                <h3>Detected Objects:</h3>
                <ul>
                  {detections.map((det) => (
                    <li key={det.id}>
                      {det.class} - Confidence: {(det.confidence * 100).toFixed(1)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
