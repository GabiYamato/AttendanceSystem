import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import VideoCapture from '../components/VideoCapture';

interface AttendanceRecord {
  student_id: string;
  student_name: string;
  timestamp: string;
  confidence: number;
  status: string;
}

const AttendancePage: React.FC = () => {
  const { classId } = useParams<{ classId: string }>();
  const [attendance, setAttendance] = useState<AttendanceRecord[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (classId) {
      fetchAttendance();
    }
  }, [classId]);

  const fetchAttendance = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/api/classes/${classId}/attendance`);
      setAttendance(response.data.attendance);
    } catch (error) {
      console.error('Error fetching attendance:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCapture = async (imageData: string) => {
    if (!isCapturing) return;

    try {
      const response = await axios.post('http://localhost:8000/api/mark-attendance', {
        class_id: classId,
        image_data: imageData
      });

      const result = response.data;
      if (result.recognized) {
        if (result.already_marked) {
          setMessage(`${result.student_name} - Already marked today`);
        } else {
          setMessage(`✅ ${result.student_name} - Attendance marked (${(result.confidence * 100).toFixed(1)}%)`);
          fetchAttendance(); // Refresh attendance list
        }
      } else {
        setMessage(`❌ ${result.message}`);
      }
    } catch (error) {
      console.error('Error marking attendance:', error);
      setMessage('Error processing attendance');
    }
  };

  const toggleCapturing = () => {
    setIsCapturing(!isCapturing);
    setMessage(isCapturing ? 'Stopped live recognition' : 'Started live recognition');
  };

  if (loading) {
    return <div className="text-center">Loading attendance data...</div>;
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-8">Attendance - Class {classId}</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Live Video Section */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Live Face Recognition</h2>
          <VideoCapture 
            onCapture={handleCapture}
            isCapturing={isCapturing}
            className="mb-4"
          />
          <div className="flex justify-between items-center">
            <button
              onClick={toggleCapturing}
              className={`px-4 py-2 rounded font-medium ${
                isCapturing 
                  ? 'bg-red-600 text-white hover:bg-red-700' 
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {isCapturing ? 'Stop Recognition' : 'Start Recognition'}
            </button>
          </div>
          {message && (
            <div className="mt-4 p-3 bg-gray-100 rounded">
              {message}
            </div>
          )}
        </div>

        {/* Attendance List */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Today's Attendance ({attendance.length})</h2>
          <div className="max-h-96 overflow-y-auto">
            {attendance.length === 0 ? (
              <p className="text-gray-500">No attendance recorded yet</p>
            ) : (
              <div className="space-y-2">
                {attendance.map((record, index) => (
                  <div key={index} className="bg-gray-50 p-3 rounded">
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">{record.student_name}</p>
                        <p className="text-sm text-gray-600">{record.student_id}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-gray-600">
                          {new Date(record.timestamp).toLocaleTimeString()}
                        </p>
                        <p className="text-xs text-gray-500">
                          {(record.confidence * 100).toFixed(1)}% confidence
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AttendancePage;
