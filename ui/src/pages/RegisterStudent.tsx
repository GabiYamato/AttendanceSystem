import React, { useState } from 'react';
import axios from 'axios';
import VideoCapture from '../components/VideoCapture';

const RegisterStudent: React.FC = () => {
  const [formData, setFormData] = useState({
    class_id: '',
    student_id: '',
    student_name: ''
  });
  const [step, setStep] = useState(1); // 1: form, 2: face capture
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await axios.post('http://localhost:8000/api/register-student', formData);
      setStep(2);
      setMessage('Student registered. Now capture face for recognition.');
    } catch (error) {
      console.error('Error registering student:', error);
      setMessage('Error registering student');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFaceCapture = async (imageData: string) => {
    setIsLoading(true);

    try {
      const response = await axios.post(
        `http://localhost:8000/api/register-face?class_id=${formData.class_id}&student_id=${formData.student_id}`,
        null,
        {
          params: {
            class_id: formData.class_id,
            student_id: formData.student_id,
            image_data: imageData
          }
        }
      );

      setMessage('✅ Face registered successfully!');
      // Reset form
      setTimeout(() => {
        setStep(1);
        setFormData({ class_id: '', student_id: '', student_name: '' });
        setMessage('');
      }, 3000);
    } catch (error: any) {
      console.error('Error registering face:', error);
      setMessage(`❌ ${error.response?.data?.detail || 'Error registering face'}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto">
      <h1 className="text-3xl font-bold mb-8 text-center">Register New Student</h1>

      {step === 1 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <form onSubmit={handleFormSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Class ID</label>
              <input
                type="text"
                name="class_id"
                value={formData.class_id}
                onChange={handleInputChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                placeholder="e.g., CS101"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Student ID</label>
              <input
                type="text"
                name="student_id"
                value={formData.student_id}
                onChange={handleInputChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                placeholder="e.g., 2024001"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Student Name</label>
              <input
                type="text"
                name="student_name"
                value={formData.student_name}
                onChange={handleInputChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                placeholder="Full Name"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? 'Registering...' : 'Next: Capture Face'}
            </button>
          </form>
        </div>
      )}

      {step === 2 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Capture Face</h2>
          <p className="text-gray-600 mb-4">
            Position your face in the camera and click capture when ready.
          </p>
          
          <VideoCapture
            onCapture={handleFaceCapture}
            isCapturing={false}
            className="mb-4"
          />

          <button
            onClick={() => setStep(1)}
            className="w-full bg-gray-600 text-white py-2 rounded hover:bg-gray-700"
          >
            Back to Form
          </button>
        </div>
      )}

      {message && (
        <div className="mt-4 p-3 bg-gray-100 rounded text-center">
          {message}
        </div>
      )}
    </div>
  );
};

export default RegisterStudent;
