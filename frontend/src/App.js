import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { Dashboard } from './pages/Dashboard';
import { StudentRegistration } from './pages/StudentRegistration';
import { AttendanceSession } from './pages/AttendanceSession';
import { AttendanceRecognition } from './pages/AttendanceRecognition';
import { ScheduleManagement } from './pages/ScheduleManagement';
import { Reports } from './pages/Reports';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/students/register" element={<StudentRegistration />} />
            <Route path="/attendance/session" element={<AttendanceSession />} />
            <Route path="/attendance/recognize" element={<AttendanceRecognition />} />
            <Route path="/schedules" element={<ScheduleManagement />} />
            <Route path="/reports" element={<Reports />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
