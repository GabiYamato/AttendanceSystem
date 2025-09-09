import { useState } from 'react';
import { Camera } from './Camera';
import { apiService } from '../services/api';
import type { StudentRegistration } from '../types';

interface StudentRegistrationProps {
  onSuccess?: () => void;
}

export const StudentRegistrationForm = ({ onSuccess }: StudentRegistrationProps) => {
  const [formData, setFormData] = useState<StudentRegistration>({
    class_id: '',
    student_id: '',
    student_name: '',
  });
  const [step, setStep] = useState<'form' | 'camera'>('form');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [studentRegistered, setStudentRegistered] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.class_id || !formData.student_id || !formData.student_name) {
      setMessage({ type: 'error', text: 'Please fill in all fields' });
      return;
    }

    setIsLoading(true);
    try {
      await apiService.registerStudent(formData);
      setStudentRegistered(true);
      setStep('camera');
      setMessage({ type: 'success', text: 'Student registered! Now capture your face for recognition.' });
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Registration failed' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFaceCapture = async (imageData: string) => {
    setIsLoading(true);
    try {
      await apiService.registerFace(formData.class_id, formData.student_id, imageData);
      setMessage({ 
        type: 'success', 
        text: 'Face registered successfully! You can now use face recognition for attendance.' 
      });
      onSuccess?.();
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Face registration failed' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({ class_id: '', student_id: '', student_name: '' });
    setStep('form');
    setMessage(null);
    setStudentRegistered(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-8 text-white text-center">
          <h2 className="text-3xl font-bold mb-2">Student Registration</h2>
          <p className="text-blue-100">
            {step === 'form' ? 'Enter your information' : 'Capture your face for recognition'}
          </p>
        </div>

        {/* Progress indicator */}
        <div className="p-6 bg-gray-50">
          <div className="flex items-center justify-center space-x-4">
            <div className={`flex items-center space-x-2 ${step === 'form' ? 'text-blue-600' : 'text-green-600'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold ${
                step === 'form' ? 'bg-blue-600 text-white' : 'bg-green-600 text-white'
              }`}>
                {studentRegistered ? '✓' : '1'}
              </div>
              <span className="font-medium">Student Info</span>
            </div>
            
            <div className="flex-1 h-1 bg-gray-300 rounded">
              <div className={`h-full rounded transition-all duration-500 ${
                step === 'camera' ? 'bg-blue-600 w-full' : 'bg-gray-300 w-0'
              }`}></div>
            </div>
            
            <div className={`flex items-center space-x-2 ${step === 'camera' ? 'text-blue-600' : 'text-gray-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold ${
                step === 'camera' ? 'bg-blue-600 text-white' : 'bg-gray-300 text-gray-600'
              }`}>
                2
              </div>
              <span className="font-medium">Face Capture</span>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-8">
          {step === 'form' && (
            <form onSubmit={handleFormSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Class ID
                </label>
                <input
                  type="text"
                  name="class_id"
                  value={formData.class_id}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="e.g., CS101, MATH202"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Student ID
                </label>
                <input
                  type="text"
                  name="student_id"
                  value={formData.student_id}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="e.g., 2023001234"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Full Name
                </label>
                <input
                  type="text"
                  name="student_name"
                  value={formData.student_name}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="e.g., John Doe"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:transform-none"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    <span>Registering...</span>
                  </div>
                ) : (
                  'Continue to Face Capture'
                )}
              </button>
            </form>
          )}

          {step === 'camera' && (
            <div className="space-y-6">
              <div className="text-center mb-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-2">
                  Capture Your Face
                </h3>
                <p className="text-gray-600">
                  Position your face within the guide and click capture when ready
                </p>
              </div>

              <Camera
                onCapture={handleFaceCapture}
                isCapturing={isLoading}
                className="mb-6"
              />

              <div className="flex space-x-4">
                <button
                  onClick={resetForm}
                  className="flex-1 py-3 border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-all"
                >
                  Start Over
                </button>
              </div>
            </div>
          )}

          {/* Message display */}
          {message && (
            <div className={`mt-6 p-4 rounded-lg ${
              message.type === 'success' 
                ? 'bg-green-100 border border-green-200 text-green-800' 
                : 'bg-red-100 border border-red-200 text-red-800'
            }`}>
              <div className="flex items-center space-x-2">
                <div className="text-lg">
                  {message.type === 'success' ? '✅' : '❌'}
                </div>
                <p className="font-medium">{message.text}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
