import React, { useState, useRef } from 'react';
import { Card, Button, Alert } from '../components/UI';
import { Camera, Upload, CheckCircle, XCircle, User } from 'lucide-react';
import Webcam from 'react-webcam';

export function AttendanceRecognition() {
  const [mode, setMode] = useState('camera'); // 'camera' or 'upload'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [alert, setAlert] = useState(null);
  const webcamRef = useRef(null);

  const capturePhoto = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    // Convert base64 to blob
    const response = await fetch(imageSrc);
    const blob = await response.blob();
    
    await processImage(blob);
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    await processImage(file);
  };

  const processImage = async (imageFile) => {
    setLoading(true);
    setResult(null);
    setAlert(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await fetch('/api/attendance/recognize', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        
        if (data.recognized) {
          setAlert({
            type: 'success',
            message: data.message
          });
        } else {
          setAlert({
            type: 'warning',
            message: data.message
          });
        }
      } else {
        throw new Error(data.detail || 'Recognition failed');
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
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold text-gray-900">Face Recognition</h1>
        <p className="text-gray-600 mt-2">
          Take a photo or upload an image for attendance
        </p>
      </div>

      {/* Mode Selection */}
      <Card>
        <div className="flex justify-center space-x-4 mb-6">
          <Button
            variant={mode === 'camera' ? 'primary' : 'secondary'}
            onClick={() => setMode('camera')}
          >
            <Camera className="h-4 w-4 mr-2" />
            Camera
          </Button>
          <Button
            variant={mode === 'upload' ? 'primary' : 'secondary'}
            onClick={() => setMode('upload')}
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Image
          </Button>
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

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Camera/Upload Section */}
          <div className="space-y-4">
            {mode === 'camera' ? (
              <div className="space-y-4">
                <div className="relative">
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    width="100%"
                    className="rounded-lg border border-gray-300"
                  />
                </div>
                <Button 
                  onClick={capturePhoto} 
                  loading={loading}
                  className="w-full"
                >
                  <Camera className="h-4 w-4 mr-2" />
                  Capture Photo
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <label htmlFor="image-upload" className="cursor-pointer">
                    <span className="text-blue-600 hover:text-blue-500 font-medium">
                      Click to upload an image
                    </span>
                    <input
                      id="image-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                    />
                  </label>
                  <p className="text-sm text-gray-500 mt-2">
                    JPG, PNG up to 10MB
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Result Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900">Recognition Result</h3>
            
            {loading ? (
              <div className="flex items-center justify-center p-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              </div>
            ) : result ? (
              <div className="space-y-4">
                {result.recognized ? (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="h-8 w-8 text-green-600" />
                      <div>
                        <h4 className="font-medium text-green-900">Student Recognized!</h4>
                        <p className="text-green-700">{result.student.name}</p>
                      </div>
                    </div>
                    
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-green-700">Student ID:</span>
                        <span className="text-sm font-medium text-green-900">{result.student.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-green-700">Confidence:</span>
                        <span className="text-sm font-medium text-green-900">
                          {(result.student.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-center space-x-3">
                      <XCircle className="h-8 w-8 text-yellow-600" />
                      <div>
                        <h4 className="font-medium text-yellow-900">No Match Found</h4>
                        <p className="text-yellow-700">Student not recognized</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center p-8 text-gray-500">
                <div className="text-center">
                  <User className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>Take a photo to start recognition</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Instructions */}
      <Card className="bg-blue-50 border-blue-200">
        <h4 className="font-medium text-blue-900 mb-3">Recognition Tips:</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
          <ul className="space-y-1">
            <li>• Ensure good lighting on your face</li>
            <li>• Look directly at the camera</li>
            <li>• Remove sunglasses and masks</li>
            <li>• Keep your face centered in frame</li>
          </ul>
          <ul className="space-y-1">
            <li>• Use a clear, high-quality image</li>
            <li>• Avoid shadows on your face</li>
            <li>• Make sure your face is fully visible</li>
            <li>• Try different angles if not recognized</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}
