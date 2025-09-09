import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import AttendancePage from './pages/AttendancePage';
import RegisterStudent from './pages/RegisterStudent';
import ClassDashboard from './pages/ClassDashboard.tsx';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App min-h-screen bg-gray-100">
        <Navbar />
        <div className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/attendance/:classId" element={<AttendancePage />} />
            <Route path="/register" element={<RegisterStudent />} />
            <Route path="/class/:classId" element={<ClassDashboard />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
