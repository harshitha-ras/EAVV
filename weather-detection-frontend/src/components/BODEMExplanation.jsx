import React, { useState, useRef } from 'react';
import { TailSpin } from "react-loader-spinner";
import './BODEMExplanation.css';

const BODEMExplanation = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [explanationUrl, setExplanationUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      // Create a preview URL for the selected image
      const imageUrl = URL.createObjectURL(file);
      setPreviewUrl(imageUrl);
      setExplanationUrl(null); // Reset any previous explanation
      setError(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const generateExplanation = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      // Assuming your API endpoint is at /api/bodem_explain
      const response = await fetch('/api/bodem_explain', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate explanation');
      }

      const data = await response.json();
      
      // Add a timestamp to prevent caching
      setExplanationUrl(`${data.explanation_path}?t=${new Date().getTime()}`);
    } catch (err) {
      setError(err.message || 'An error occurred while generating the explanation');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bodem-container">
      <h2>BODEM Explanation Visualizer</h2>
      <p className="description">
        Upload an image to see how the model detects objects in various weather conditions.
      </p>

      <div className="upload-section">
        <button 
          className="upload-button" 
          onClick={handleUploadClick}
        >
          Select Image
        </button>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleImageChange}
          accept="image/*"
          style={{ display: 'none' }}
        />
        <button
          className="generate-button"
          onClick={generateExplanation}
          disabled={!selectedImage || isLoading}
        >
          {isLoading ? 'Generating...' : 'Generate BODEM Explanation'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="preview-container">
        {previewUrl && (
          <div className="image-preview">
            <h3>Original Image</h3>
            <img src={previewUrl} alt="Preview" />
          </div>
        )}
        
        {isLoading && (
          <div className="loading-container">
            <TailSpin
              color="#4285f4"
              height={80}
              width={80}
              radius={1}
              visible={true}
            />
            <p>Generating BODEM explanation... This may take a moment.</p>
          </div>
        )}
        
        {explanationUrl && !isLoading && (
          <div className="explanation-preview">
            <h3>BODEM Explanation</h3>
            <img src={explanationUrl} alt="BODEM Explanation" />
            <div className="explanation-legend">
              <p><strong>Left:</strong> Original image with detections</p>
              <p><strong>Right:</strong> Saliency map showing important regions for detection</p>
              <p>Brighter areas indicate regions that influenced the model's decisions more strongly.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BODEMExplanation;
