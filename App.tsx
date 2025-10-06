import React from 'react';
import HandDetector from './components/HandDetector';

const App: React.FC = () => {
  return (
    <div className="w-screen h-screen bg-black text-gray-200">
      <HandDetector />
    </div>
  );
};

export default App;
