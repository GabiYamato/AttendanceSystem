import React, { useState } from 'react';
import { FaceCapture } from './FaceCapture';
import { apiService } from '../services/api';
import { User, BookOpen } from 'lucide-react';

export const StudentRegistration: React.FC = () => {
  const [step, setStep] = useState<'details' | 'face'>('details');
  const [formData, setFormData] = useState({
    classId: '',
    studentId: '',
    studentName: '',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleDetailsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.classId || !formData.studentId || !formData.studentName) {
      setMessage('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setMessage('');

    try {
      await apiService.registerStudent({
        class_id: formData.classId,
        student_id: formData.studentId,
        student_name: formData.studentName,
      });
      setStep('face');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFaceCapture = async (imageData: string) => {
    setIsLoading(true);
    setMessage('');

    try {
      await apiService.registerFace(formData.classId, formData.studentId, imageData);
      setMessage('Student registered successfully!');
      setTimeout(() => {
        setStep('details');
        setFormData({ classId: '', studentId: '', studentName: '' });
        setMessage('');
      }, 3000);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Face registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  if (step === 'face') {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-center mb-2">Register Face</h2>
          <p className="text-gray-600 text-center mb-8">
            Step 2: Capture face data for <strong>{formData.studentName}</strong>
          </p>
          
          <FaceCapture
            onCapture={handleFaceCapture}
            onError={setMessage}
            isProcessing={isLoading}
            className="mb-6"
          />
          
          {message && (
            <div className={`p-4 rounded-lg text-center ${
              message.includes('success') 
                ? 'bg-green-100 text-green-700' 
                : 'bg-red-100 text-red-700'
            }`}>
              {message}
            </div>
          )}
          
          <button
            onClick={() => setStep('details')}
            className="w-full mt-4 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Back to Details
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-md mx-auto p-6">
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="text-center mb-8">
          <User className="w-16 h-16 mx-auto text-blue-500 mb-4" />
          <h2 className="text-2xl font-bold">Register Student</h2>
          <p className="text-gray-600">Step 1: Enter student details</p>
        </div>

        <form onSubmit={handleDetailsSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <BookOpen className="inline w-4 h-4 mr-2" />
              Class ID
            </label>
            <input
              type="text"
              value={formData.classId}
              onChange={(e) => setFormData(prev => ({ ...prev, classId: e.target.value }))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g., CS101"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Student ID
            </label>
            <input
              type="text"
              value={formData.studentId}
              onChange={(e) => setFormData(prev => ({ ...prev, studentId: e.target.value }))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g., 2023001"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Student Name
            </label>
            <input
              type="text"
              value={formData.studentName}
              onChange={(e) => setFormData(prev => ({ ...prev, studentName: e.target.value }))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="e.g., John Doe"
              required
            />
          </div>

          {message && (
            <div className="p-4 bg-red-100 text-red-700 rounded-lg text-center">
              {message}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white font-medium rounded-lg transition-colors duration-200"
          >
            {isLoading ? 'Registering...' : 'Continue to Face Registration'}
          </button>
        </form>
      </div>
    </div>
  );
};
