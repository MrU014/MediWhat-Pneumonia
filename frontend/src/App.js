import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
    console.log('File selected:', e.target.files[0]?.name);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submit clicked, file:', file?.name);
    
    if (!file) {
      setError('Please select a file');
      console.log('No file error');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Sending request to backend...');
      
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      console.log('API Response received:', response.data);
      
      setResult(response.data);
      setError(null);
    } catch (err) {
      console.error('API Error details:', err);
      console.log('Error response:', err.response?.data);
      setError(err.response?.data?.error || 'An error occurred');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Pneumonia Detection from Chest X-Ray</h1>
        <p>Upload a chest X-ray image (JPG/PNG) to detect pneumonia with explainable AI analysis.</p>
        
        <form onSubmit={handleSubmit}>
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !file}>
            {loading ? 'Analyzing...' : 'Detect'}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <p style={{ 
              fontWeight: '600', 
              fontSize: '1.2em',
              color: result.diagnosis === 'PNEUMONIA' ? '#e74c3c' : '#27ae60'
            }}>
              Pneumonia Detected: {result.diagnosis === 'PNEUMONIA' ? 'Yes' : 'No'}
            </p>
            <p>Confidence: {result.confidence}</p>

            {result.explanation && (
              <div style={{
                marginTop: '20px',
                padding: '20px',
                background: '#f0f8ff',
                borderLeft: '4px solid #3498db',
                borderRadius: '8px',
                textAlign: 'left'
              }}>
                <h4 style={{ 
                  fontSize: '1.1em', 
                  color: '#2c3e50', 
                  marginBottom: '12px',
                  marginTop: '0'
                }}>
                  📋 AI Analysis
                </h4>
                <p style={{ 
                  fontSize: '0.95em', 
                  lineHeight: '1.8', 
                  color: '#555',
                  margin: '0',
                  whiteSpace: 'pre-wrap'
                }}>
                  {result.explanation}
                </p>
              </div>
            )}

            {result.heatmap_image ? (
              <>
                <p style={{ marginTop: '20px' }}>
                  Explanation Heatmap (colored areas show regions influencing the AI's decision):
                </p>
                <img 
                  src={result.heatmap_image} 
                  alt="GRAD-CAM Heatmap" 
                  style={{ 
                    maxWidth: '100%', 
                    height: 'auto', 
                    border: '2px solid #e9ecef',
                    borderRadius: '8px',
                    marginTop: '10px'
                  }} 
                  onLoad={() => console.log('Heatmap loaded successfully!')}
                  onError={(e) => console.error('Heatmap load failed:', e)}
                />
              </>
            ) : (
              <p style={{ color: 'orange', marginTop: '20px' }}>
                Heatmap visualization unavailable (check console for details).
              </p>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;