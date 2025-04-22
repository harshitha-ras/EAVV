import React, { useState, useRef } from 'react';
import './BODEMExplanation.css';

// Icons (using inline SVG for simplicity)
const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 16L12 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M9 11L12 8 15 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M8 16H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M3 20H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const ExplainIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 16C14.2091 16 16 14.2091 16 12C16 9.79086 14.2091 8 12 8C9.79086 8 8 9.79086 8 12C8 14.2091 9.79086 16 12 16Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M12 2V4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M12 20V22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M4.93 4.93L6.34 6.34" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M17.66 17.66L19.07 19.07" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M2 12H4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M20 12H22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M6.34 17.66L4.93 19.07" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M19.07 4.93L17.66 6.34" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const FileIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="file-info-icon">
    <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M14 2V8H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const CloseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const WarningIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 9V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M12 17H12.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M10.29 3.86L1.82 18C1.64537 18.3024 1.55296 18.6453 1.55199 18.9945C1.55101 19.3437 1.6415 19.6871 1.81442 19.9905C1.98734 20.2939 2.23672 20.5467 2.53773 20.7238C2.83875 20.9009 3.18058 20.9962 3.53 21H20.47C20.8194 20.9962 21.1613 20.9009 21.4623 20.7238C21.7633 20.5467 22.0127 20.2939 22.1856 19.9905C22.3585 19.6871 22.449 19.3437 22.448 18.9945C22.447 18.6453 22.3546 18.3024 22.18 18L13.71 3.86C13.5317 3.56611 13.2807 3.32312 12.9812 3.15448C12.6817 2.98585 12.3437 2.89725 12 2.89725C11.6563 2.89725 11.3183 2.98585 11.0188 3.15448C10.7193 3.32312 10.4683 3.56611 10.29 3.86Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// Loading spinner component
const LoadingSpinner = () => (
  <div className="spinner">
    <svg width="50" height="50" viewBox="0 0 50 50">
      <circle cx="25" cy="25" r="20" fill="none" stroke="#3498db" strokeWidth="5" strokeLinecap="round" strokeDasharray="31.415, 31.415" strokeDashoffset="0">
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 25 25"
          to="360 25 25"
          dur="1s"
          repeatCount="indefinite"
        />
      </circle>
    </svg>
  </div>
);

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

  const clearSelectedFile = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
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
      <div className="bodem-header">
        <div className="bodem-icon">
          <ExplainIcon />
        </div>
        <h2>BODEM Explanation Visualizer</h2>
      </div>
      
      <p className="description">
        Upload an image to visualize how the model detects objects in various weather conditions. 
        BODEM helps explain which parts of the image influence the model's detection decisions.
      </p>

      <div className="upload-section">
        <button 
          className="upload-button" 
          onClick={handleUploadClick}
        >
          <UploadIcon /> Select Image
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

      {selectedImage && (
        <div className="file-info">
          <FileIcon />
          <span className="file-name">{selectedImage.name}</span>
          <button className="clear-file" onClick={clearSelectedFile} title="Remove file">
            <CloseIcon />
          </button>
        </div>
      )}

      {error && (
        <div className="error-message">
          <WarningIcon />
          {error}
        </div>
      )}

      <div className="preview-container">
        {previewUrl && (
          <div className="image-preview">
            <h3>Original Image</h3>
            <img src={previewUrl} alt="Preview" />
          </div>
        )}
        
        {isLoading && (
          <div className="loading-container">
            <LoadingSpinner />
            <p>Generating BODEM explanation...<br />This may take a moment as we analyze the image.</p>
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
