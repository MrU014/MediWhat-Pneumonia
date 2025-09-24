import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
    console.log('File selected:', e.target.files[0]?.name);  // Extra log: Confirms file pick
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submit clicked, file:', file?.name);  // Extra log: Confirms button press
    
    if (!file) {
      setError('Please select a file');
      console.log('No file error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Sending request to backend...');  // Extra log: Before API call
      
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      console.log('API Response received:', response.data);  // Main log: Full response
      
      setResult(response.data);
      setError(null);
    } catch (err) {
      console.error('API Error details:', err);  // Extra log: Full error if fails
      console.log('Error response:', err.response?.data);  // If backend error
      setError(err.response?.data?.error || 'An error occurred');
      setResult(null);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Pneumonia Detection from Chest X-Ray</h1>
        <p>Upload a chest X-ray image (JPG/PNG) to detect pneumonia (yes/no) with explainable heatmap.</p>
        <form onSubmit={handleSubmit}>
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <button type="submit">Detect</button>
        </form>
        {error && <p className="error">{error}</p>}
        {result && (
          <div className="result">
            <p>Pneumonia Detected: {result.diagnosis === 'PNEUMONIA' ? 'Yes' : 'No'}</p>
            <p>Confidence: {result.confidence}</p>
            {result.heatmap_image ? (
              <>
                <p>Explanation Heatmap (red areas indicate influencing regions):</p>
                <img 
                  src={result.heatmap_image} 
                  alt="GRAD-CAM Heatmap" 
                  style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ccc' }} 
                  onLoad={() => console.log('Image loaded successfully!')}  // Extra: Confirms load
                  onError={(e) => console.error('Image load failed:', e)}  // Extra: Broken image log
                />
              </>
            ) : (
              <p style={{ color: 'orange' }}>Heatmap unavailable (check console for details).</p>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;