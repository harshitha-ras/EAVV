import React from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Weather Condition Object Detection</h1>
      </header>
      <main>
        <ImageUpload />
      </main>
    </div>
  );
}

export default App;

