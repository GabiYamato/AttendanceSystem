import React, { useState } from 'react';
import { Card, Button, Alert } from '../components/UI';
import { Play, QrCode, Users, Clock, Download } from 'lucide-react';
import QRCodeReact from 'qrcode.react';

export function AttendanceSession() {
  const [sessionData, setSessionData] = useState({
    class_id: '',
    session_name: '',
    duration_minutes: 60
  });
  const [activeSession, setActiveSession] = useState(null);
  const [loading, setLoading] = useState(false);
  const [alert, setAlert] = useState(null);

  const handleInputChange = (e) => {
    setSessionData({
      ...sessionData,
      [e.target.name]: e.target.value
    });
  };

  const startSession = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAlert(null);

    try {
      const response = await fetch('/api/attendance/start-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sessionData)
      });

      const data = await response.json();

      if (data.success) {
        setActiveSession({
          session_id: data.session_id,
          qr_code: data.qr_code,
          students_loaded: data.students_loaded,
          expires_at: data.expires_at,
          class_id: sessionData.class_id,
          session_name: sessionData.session_name
        });
        
        setAlert({
          type: 'success',
          message: `Session started successfully! ${data.students_loaded} students loaded.`
        });
      } else {
        throw new Error(data.detail || 'Failed to start session');
      }
    } catch (error) {
      setAlert({
        type: 'error',
        message: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const exportAttendance = async () => {
    if (!activeSession) return;

    try {
      const response = await fetch(`/api/attendance/export/${activeSession.session_id}`);
      const blob = await response.blob();
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `attendance_${activeSession.session_id}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      setAlert({
        type: 'error',
        message: 'Failed to export attendance'
      });
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold text-gray-900">Attendance Session</h1>
        <p className="text-gray-600 mt-2">
          Start a new attendance session for your class
        </p>
      </div>

      {/* Alert */}
      {alert && (
        <Alert 
          type={alert.type} 
          onClose={() => setAlert(null)}
        >
          {alert.message}
        </Alert>
      )}

      {!activeSession ? (
        /* Session Setup Form */
        <Card>
          <form onSubmit={startSession} className="space-y-6">
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <Play className="h-5 w-5 mr-2" />
              Create New Session
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Class ID
                </label>
                <input
                  type="text"
                  name="class_id"
                  value={sessionData.class_id}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="e.g., CS-A, ECE-B"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Session Name
                </label>
                <input
                  type="text"
                  name="session_name"
                  value={sessionData.session_name}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="e.g., Morning Lecture, Lab Session"
                  required
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Duration (minutes)
              </label>
              <select
                name="duration_minutes"
                value={sessionData.duration_minutes}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value={30}>30 minutes</option>
                <option value={45}>45 minutes</option>
                <option value={60}>1 hour</option>
                <option value={90}>1.5 hours</option>
                <option value={120}>2 hours</option>
                <option value={180}>3 hours</option>
              </select>
            </div>

            <div className="flex justify-end">
              <Button type="submit" loading={loading}>
                <Play className="h-4 w-4 mr-2" />
                Start Session
              </Button>
            </div>
          </form>
        </Card>
      ) : (
        /* Active Session Display */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Session Info */}
          <Card>
            <div className="text-center space-y-4">
              <div className="flex items-center justify-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-lg font-medium text-gray-900">Session Active</span>
              </div>
              
              <div className="space-y-2">
                <h3 className="text-xl font-bold text-gray-900">{activeSession.session_name}</h3>
                <p className="text-gray-600">Class: {activeSession.class_id}</p>
                <p className="text-sm text-gray-500">
                  Session ID: {activeSession.session_id}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="bg-blue-50 rounded-lg p-3">
                  <Users className="h-6 w-6 text-blue-600 mx-auto mb-1" />
                  <p className="text-sm text-gray-600">Students Loaded</p>
                  <p className="text-lg font-bold text-blue-600">{activeSession.students_loaded}</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3">
                  <Clock className="h-6 w-6 text-green-600 mx-auto mb-1" />
                  <p className="text-sm text-gray-600">Expires At</p>
                  <p className="text-sm font-medium text-green-600">
                    {new Date(activeSession.expires_at).toLocaleTimeString()}
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <Button 
                  variant="secondary" 
                  className="w-full"
                  onClick={() => window.open('/attendance/recognize', '_blank')}
                >
                  Open Face Recognition
                </Button>
                <Button 
                  variant="secondary" 
                  className="w-full"
                  onClick={exportAttendance}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export Attendance
                </Button>
                <Button 
                  variant="danger" 
                  className="w-full"
                  onClick={() => setActiveSession(null)}
                >
                  End Session
                </Button>
              </div>
            </div>
          </Card>

          {/* QR Code */}
          <Card>
            <div className="text-center space-y-4">
              <div className="flex items-center justify-center space-x-2">
                <QrCode className="h-5 w-5 text-gray-600" />
                <span className="text-lg font-medium text-gray-900">QR Code</span>
              </div>
              
              <div className="flex justify-center">
                <div className="bg-white p-4 rounded-lg border-2 border-gray-200">
                  <QRCodeReact
                    value={`data:image/png;base64,${activeSession.qr_code}`}
                    size={200}
                    level="H"
                  />
                </div>
              </div>
              
              <div className="text-sm text-gray-600">
                <p>Students can scan this QR code to</p>
                <p>access the attendance portal</p>
              </div>

              <Button variant="secondary" className="w-full">
                <Download className="h-4 w-4 mr-2" />
                Download QR Code
              </Button>
            </div>
          </Card>
        </div>
      )}

      {/* Instructions */}
      <Card className="bg-blue-50 border-blue-200">
        <h4 className="font-medium text-blue-900 mb-3">How to use:</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
          <div>
            <h5 className="font-medium mb-2">For Teachers:</h5>
            <ul className="space-y-1">
              <li>• Start a session for your class</li>
              <li>• Share the QR code with students</li>
              <li>• Monitor attendance in real-time</li>
              <li>• Export attendance records</li>
            </ul>
          </div>
          <div>
            <h5 className="font-medium mb-2">For Students:</h5>
            <ul className="space-y-1">
              <li>• Scan the QR code with your phone</li>
              <li>• Take a clear photo of your face</li>
              <li>• System will automatically mark attendance</li>
              <li>• Get instant confirmation</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}
