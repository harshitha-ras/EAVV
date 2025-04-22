import React from 'react';
import './App.css';
import BODEMExplanation from './components/BODEMExplanation';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Weather Condition Object Detection</h1>
        <p>Advanced detection in adverse weather using YOLOv8</p>
      </header>
      
      <main>
        <BODEMExplanation />
      </main>
      
      <footer className="App-footer">
        <p>DAWN-WEDGE Weather Detection Project &copy; 2025</p>
      </footer>
    </div>
  );
}

export default App;
