import './ImageUpload.css';
import React, { useState } from 'react';

const ImageUpload = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    // Display image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
    
    // Create FormData for API request
    const formData = new FormData();
    formData.append("image", file);
    
    setIsLoading(true);
    setErrorMessage(null);
    
    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      setDetectionResults(data);
    } catch (error) {
      console.error("Failed to fetch:", error);
      setErrorMessage("Failed to process image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="image-upload-container">
      <h2>Upload an Image for Weather Condition Detection</h2>
      
      <div className="upload-area">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload} 
          id="image-input"
        />
        <label htmlFor="image-input" className="upload-button">
          Select Image
        </label>
      </div>
      
      {errorMessage && (
        <div className="error-message">
          {errorMessage}
        </div>
      )}
      
      {isLoading && <div className="loading">Processing image...</div>}
      
      {imagePreview && (
        <div className="preview-container">
          <h3>Image Preview</h3>
          <img src={imagePreview} alt="Preview" className="image-preview" />
        </div>
      )}
      
      {detectionResults && (
        <div className="results-container">
          <h3>Detection Results</h3>
          <pre>{JSON.stringify(detectionResults, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;

