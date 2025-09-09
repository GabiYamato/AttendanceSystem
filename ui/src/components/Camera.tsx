import { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';

interface CameraProps {
  onCapture: (imageData: string) => void;
  isCapturing?: boolean;
  className?: string;
}

export const Camera = ({ onCapture, isCapturing = false, className = '' }: CameraProps) => {
  const webcamRef = useRef<Webcam>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: 'user',
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      onCapture(imageSrc);
    }
  }, [onCapture]);

  const handleUserMedia = () => {
    setIsCameraReady(true);
    setError(null);
  };

  const handleUserMediaError = (error: string | DOMException) => {
    console.error('Camera error:', error);
    setError('Unable to access camera. Please check permissions.');
    setIsCameraReady(false);
  };

  return (
    <div className={`relative ${className}`}>
      <div className="relative overflow-hidden rounded-xl shadow-2xl bg-gray-900">
        {!isCameraReady && !error && (
          <div className="absolute inset-0 bg-gray-800 flex items-center justify-center z-10">
            <div className="text-center text-white">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
              <p>Starting camera...</p>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 bg-red-900 flex items-center justify-center z-10">
            <div className="text-center text-white p-6">
              <div className="text-4xl mb-4">📷</div>
              <p className="text-lg">{error}</p>
              <p className="text-sm mt-2 opacity-75">
                Please allow camera access and refresh the page
              </p>
            </div>
          </div>
        )}

        <Webcam
          ref={webcamRef}
          audio={false}
          height={480}
          width={640}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          onUserMedia={handleUserMedia}
          onUserMediaError={handleUserMediaError}
          className="w-full h-auto"
        />
        
        {/* Face detection overlay */}
        {isCameraReady && (
          <div className="absolute inset-0 pointer-events-none">
            {/* Face outline guide */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-48 h-64 border-2 border-white border-opacity-50 rounded-full flex items-center justify-center">
                <div className="text-white text-sm opacity-75 text-center">
                  Position your face<br />within this area
                </div>
              </div>
            </div>
            
            {/* Scan animation when capturing */}
            {isCapturing && (
              <div className="absolute inset-0 bg-blue-500 bg-opacity-20 animate-pulse">
                <div className="absolute top-0 left-0 right-0 h-1 bg-blue-400 animate-bounce"></div>
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Capture button */}
      <div className="mt-6 text-center">
        <button
          onClick={capture}
          disabled={!isCameraReady || isCapturing}
          className={`px-8 py-4 rounded-full font-semibold text-lg transition-all duration-200 transform ${
            isCameraReady && !isCapturing
              ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:scale-105 active:scale-95'
              : 'bg-gray-400 text-gray-200 cursor-not-allowed'
          }`}
        >
          {isCapturing ? (
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Processing...</span>
            </div>
          ) : (
            '📸 Capture Face'
          )}
        </button>
      </div>
    </div>
  );
};
