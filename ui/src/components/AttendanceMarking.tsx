import { useState } from 'react';
import { Camera } from './Camera';
import { apiService } from '../services/api';
import type { FaceRecognitionResponse } from '../types';

export const AttendanceMarking = () => {
  const [classId, setClassId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<FaceRecognitionResponse | null>(null);

  const handleFaceCapture = async (imageData: string) => {
    if (!classId.trim()) {
      setResult({
        recognized: false,
        message: 'Please enter a class ID first'
      });
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const response = await apiService.markAttendance(classId, imageData);
      setResult(response);
    } catch (error) {
      setResult({
        recognized: false,
        message: error instanceof Error ? error.message : 'Attendance marking failed',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const resetResult = () => {
    setResult(null);
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-green-600 to-blue-600 p-8 text-white text-center">
          <h2 className="text-3xl font-bold mb-2">Mark Attendance</h2>
          <p className="text-green-100">
            Scan your face to mark your attendance
          </p>
        </div>

        <div className="p-8">
          {/* Class ID Input */}
          <div className="mb-8">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Class ID
            </label>
            <input
              type="text"
              value={classId}
              onChange={(e) => setClassId(e.target.value)}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all"
              placeholder="e.g., CS101, MATH202"
              required
            />
          </div>

          {/* Camera Section */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">
              Face Recognition Scanner
            </h3>
            <Camera
              onCapture={handleFaceCapture}
              isCapturing={isLoading}
            />
          </div>

          {/* Result Display */}
          {result && (
            <div className={`p-6 rounded-xl border-2 ${
              result.recognized 
                ? 'bg-green-50 border-green-200' 
                : 'bg-red-50 border-red-200'
            }`}>
              <div className="text-center">
                <div className="text-6xl mb-4">
                  {result.recognized ? '✅' : '❌'}
                </div>
                
                {result.recognized && result.student_name && (
                  <h3 className="text-2xl font-bold text-green-800 mb-2">
                    Welcome, {result.student_name}!
                  </h3>
                )}
                
                <p className={`text-lg font-medium mb-4 ${
                  result.recognized ? 'text-green-700' : 'text-red-700'
                }`}>
                  {result.message}
                </p>
                
                {result.confidence && (
                  <div className="mb-4">
                    <p className="text-sm text-gray-600 mb-2">Recognition Confidence</p>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-500 ${
                          result.confidence > 0.8 ? 'bg-green-500' : 
                          result.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${result.confidence * 100}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                
                {result.already_marked && (
                  <div className="bg-yellow-100 border border-yellow-200 rounded-lg p-3 mb-4">
                    <p className="text-yellow-800 text-sm font-medium">
                      ⚠️ You have already been marked present today
                    </p>
                  </div>
                )}

                <button
                  onClick={resetResult}
                  className="px-6 py-2 bg-gray-600 text-white font-semibold rounded-lg hover:bg-gray-700 transition-all"
                >
                  Scan Again
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
