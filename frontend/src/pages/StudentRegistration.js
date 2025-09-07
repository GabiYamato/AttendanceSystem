import React, { useState } from 'react';
import { Card, Button, Alert } from '../components/UI';
import { Upload, Camera, User, Mail, Hash } from 'lucide-react';

export function StudentRegistration() {
  const [formData, setFormData] = useState({
    name: '',
    roll_no: '',
    email: '',
    class_id: ''
  });
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [alert, setAlert] = useState(null);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      setImages(files);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (images.length < 3) {
      setAlert({
        type: 'error',
        message: 'Please upload at least 3 images for better accuracy'
      });
      return;
    }

    setLoading(true);
    setAlert(null);

    try {
      const formPayload = new FormData();
      formPayload.append('name', formData.name);
      formPayload.append('roll_no', formData.roll_no);
      formPayload.append('email', formData.email);
      formPayload.append('class_id', formData.class_id);
      
      images.forEach((image) => {
        formPayload.append('images', image);
      });

      const response = await fetch('/api/students/register', {
        method: 'POST',
        body: formPayload
      });

      const data = await response.json();

      if (data.success) {
        setAlert({
          type: 'success',
          message: `Student ${formData.name} registered successfully with ${data.embeddings_processed} face embeddings!`
        });
        
        // Reset form
        setFormData({
          name: '',
          roll_no: '',
          email: '',
          class_id: ''
        });
        setImages([]);
        document.getElementById('images').value = '';
      } else {
        throw new Error(data.detail || 'Registration failed');
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

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold text-gray-900">Student Registration</h1>
        <p className="text-gray-600 mt-2">
          Register a new student with face recognition data
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

      {/* Registration Form */}
      <Card>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Personal Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <User className="h-5 w-5 mr-2" />
              Personal Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter student's full name"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Hash className="h-4 w-4 inline mr-1" />
                  Roll Number
                </label>
                <input
                  type="text"
                  name="roll_no"
                  value={formData.roll_no}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="e.g., CS-2024-001"
                  required
                />
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Mail className="h-4 w-4 inline mr-1" />
                  Email Address
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="student@example.com"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Class ID
                </label>
                <input
                  type="text"
                  name="class_id"
                  value={formData.class_id}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="e.g., CS-A, ECE-B"
                  required
                />
              </div>
            </div>
          </div>

          {/* Face Images */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <Camera className="h-5 w-5 mr-2" />
              Face Images
            </h3>
            
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
              <div className="text-center">
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <label htmlFor="images" className="cursor-pointer">
                  <span className="text-blue-600 hover:text-blue-500 font-medium">
                    Click to upload images
                  </span>
                  <span className="text-gray-600"> or drag and drop</span>
                </label>
                <input
                  id="images"
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  required
                />
                <p className="text-sm text-gray-500 mt-2">
                  Upload at least 3-5 clear face images (JPG, PNG)
                </p>
              </div>
            </div>
            
            {images.length > 0 && (
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-700 mb-2">
                  Selected Images ({images.length})
                </p>
                <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
                  {images.map((image, index) => (
                    <div key={index} className="relative">
                      <img
                        src={URL.createObjectURL(image)}
                        alt={`Face ${index + 1}`}
                        className="w-full h-20 object-cover rounded-lg border border-gray-200"
                      />
                      <div className="absolute top-1 right-1 bg-blue-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                        {index + 1}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-medium text-blue-900 mb-2">Photography Guidelines:</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Ensure good lighting and clear face visibility</li>
              <li>• Include different angles (front, slight left, slight right)</li>
              <li>• Avoid sunglasses, masks, or face coverings</li>
              <li>• Use high-quality images (minimum 300x300 pixels)</li>
              <li>• Include at least 3-5 images for better accuracy</li>
            </ul>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end space-x-3">
            <Button 
              type="button" 
              variant="secondary"
              onClick={() => {
                setFormData({
                  name: '',
                  roll_no: '',
                  email: '',
                  class_id: ''
                });
                setImages([]);
                document.getElementById('images').value = '';
                setAlert(null);
              }}
            >
              Reset Form
            </Button>
            <Button 
              type="submit" 
              loading={loading}
              disabled={images.length < 3}
            >
              Register Student
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
}
