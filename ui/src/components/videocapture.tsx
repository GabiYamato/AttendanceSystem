import React, { useRef, useEffect, useState } from 'react';

interface VideoCaptureProps {
  onCapture: (imageData: string) => void;
  isCapturing: boolean;
  className?: string;
}

const VideoCapture: React.FC<VideoCaptureProps> = ({ onCapture, isCapturing, className = "" }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (isCapturing) {
      const interval = setInterval(() => {
        captureFrame();
      }, 1000); // Capture every second for attendance

      return () => clearInterval(interval);
    }
  }, [isCapturing]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        } 
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      if (context) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        onCapture(imageData);
      }
    }
  };

  const manualCapture = () => {
    captureFrame();
  };

  return (
    <div className={`relative ${className}`}>
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted
        className="w-full rounded-lg border-2 border-gray-300"
      />
      <canvas 
        ref={canvasRef} 
        className="hidden"
      />
      <button
        onClick={manualCapture}
        className="absolute bottom-4 right-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Capture
      </button>
      {isCapturing && (
        <div className="absolute top-4 left-4 bg-green-500 text-white px-3 py-1 rounded">
          Live Recognition Active
        </div>
      )}
    </div>
  );
};

export default VideoCapture;
