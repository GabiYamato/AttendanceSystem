import React, { useState, useCallback } from 'react'
import * as faceapi from 'face-api.js'
import { useFaceAPI } from '../hooks/useFaceAPI'
import { useCamera } from '../hooks/useCamera'
import type { RegisteredFace } from '../types'
import {
  saveRegisteredFace
} from '../services/firebaseService'

interface RegistrationPageProps {
  registeredFaces: RegisteredFace[]
  onFaceRegistered: (face: RegisteredFace) => void
  onBack: () => void
}

const RegistrationPage: React.FC<RegistrationPageProps> = ({
  registeredFaces,
  onFaceRegistered,
  onBack
}) => {
  const { isModelLoaded, modelError } = useFaceAPI()
  const { 
    cameraActive, 
    cameraError, 
    videoRef, 
    canvasRef, 
    startCamera, 
    stopCamera, 
    setupCanvas 
  } = useCamera()
  
  const [registerName, setRegisterName] = useState('')
  const [status, setStatus] = useState('')
  const [isRegistering, setIsRegistering] = useState(false)

  const handleRegisterFace = useCallback(async () => {
    if (!registerName.trim()) {
      setStatus('Please enter a name')
      return
    }

    if (!isModelLoaded) {
      setStatus('Models are still loading...')
      return
    }

    const cameraStarted = await startCamera()
    if (!cameraStarted) {
      setStatus('Failed to start camera')
      return
    }

    setIsRegistering(true)
    setStatus('Position your face in the camera and wait...')
    console.log('Starting face registration for:', registerName)

    // Start the registration process after a short delay
    setTimeout(startRegistrationProcess, 1000)
  }, [registerName, isModelLoaded, startCamera])

  const startRegistrationProcess = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    await setupCanvas()

    let detectionCount = 0
    const maxDetections = 5
    const descriptors: Float32Array[] = []

    const detectForRegistration = async () => {
      if (detectionCount >= maxDetections) return

      try {
        const detection = await faceapi
          .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ 
            inputSize: 416, 
            scoreThreshold: 0.5 
          }))
          .withFaceLandmarks()
          .withFaceDescriptor()

        ctx.clearRect(0, 0, canvas.width, canvas.height)

        if (detection) {
          const box = detection.detection.box
          
          // Draw bounding box
          ctx.strokeStyle = '#27ae60'
          ctx.lineWidth = 3
          ctx.strokeRect(box.x, box.y, box.width, box.height)
          
          // Draw background for text
          ctx.fillStyle = 'rgba(39, 174, 96, 0.8)'
          ctx.fillRect(box.x, box.y - 35, box.width, 35)
          
          // Draw label
          ctx.fillStyle = '#ffffff'
          ctx.font = 'bold 16px Arial'
          ctx.textAlign = 'center'
          ctx.fillText(
            `Registering: ${registerName} (${detectionCount + 1}/${maxDetections})`,
            box.x + box.width / 2,
            box.y - 10
          )
          ctx.textAlign = 'start'

          detectionCount++
          descriptors.push(detection.descriptor)
          console.log(`Face detected for registration (${detectionCount}/${maxDetections})`)

          if (detectionCount >= maxDetections) {
            await completeRegistration(descriptors)
            return
          }
        } else {
          console.log('No face detected during registration')
          // Draw message when no face is detected
          ctx.fillStyle = 'rgba(231, 76, 60, 0.8)'
          ctx.fillRect(20, 20, canvas.width - 40, 60)
          ctx.fillStyle = '#ffffff'
          ctx.font = 'bold 18px Arial'
          ctx.textAlign = 'center'
          ctx.fillText('Please position your face in the camera', canvas.width / 2, 55)
          ctx.textAlign = 'start'
        }
      } catch (error) {
        console.error('Error in registration detection:', error)
      }

      setTimeout(detectForRegistration, 200)
    }

    detectForRegistration()
  }, [registerName, setupCanvas])

  const completeRegistration = useCallback(async (descriptors: Float32Array[]) => {
    try {
      // Average the descriptors for better accuracy
      const avgDescriptor = new Float32Array(descriptors[0].length)
      for (let i = 0; i < avgDescriptor.length; i++) {
        let sum = 0
        for (const desc of descriptors) {
          sum += desc[i]
        }
        avgDescriptor[i] = sum / descriptors.length
      }

      const newFace: RegisteredFace = {
        id: `face-${Date.now()}`,
        name: registerName.trim(),
        descriptor: avgDescriptor,
        timestamp: new Date().toISOString()
      }

      // Save to Firebase
      await saveRegisteredFace({
        ...newFace,
        descriptor: Array.from(avgDescriptor)
      })
      
      onFaceRegistered(newFace)
      setStatus(`Successfully registered ${registerName}!`)
      console.log('Face successfully registered:', newFace.name)
      
      setRegisterName('')
      setIsRegistering(false)
      
      setTimeout(() => {
        stopCamera()
        setStatus('')
        onBack()
      }, 2000)
    } catch (error) {
      console.error('Error saving face:', error)
      setStatus('Error saving face. Please try again.')
      setIsRegistering(false)
    }
  }, [registerName, onFaceRegistered, stopCamera, onBack])

  const handleStop = useCallback(() => {
    setIsRegistering(false)
    stopCamera()
    setStatus('')
  }, [stopCamera])

  return (
    <div className="register-view">
      <div className="view-header">
        <h2>Register New Face</h2>
        <p className="view-description">
          Enter a name and position the face in the camera to register
        </p>
      </div>

      <div className="register-form">
        <div className="form-group">
          <label>Name:</label>
          <input
            type="text"
            value={registerName}
            onChange={(e) => setRegisterName(e.target.value)}
            placeholder="Enter person's name"
            disabled={isRegistering}
          />
        </div>
        
        <div className="form-actions">
          {!isRegistering ? (
            <button 
              onClick={handleRegisterFace}
              className="register-face-btn"
              disabled={!registerName.trim() || !isModelLoaded}
            >
              {!isModelLoaded ? 'Loading Models...' : 'Start Registration'}
            </button>
          ) : (
            <button 
              onClick={handleStop}
              className="stop-btn"
            >
              Stop Registration
            </button>
          )}
          
          <button 
            onClick={onBack}
            className="back-btn"
            disabled={isRegistering}
          >
            Back to Dashboard
          </button>
        </div>
        
        {(status || modelError || cameraError) && (
          <div className={`status-message ${modelError || cameraError ? 'error' : ''}`}>
            {modelError || cameraError || status}
          </div>
        )}
      </div>

      <div className="camera-container">
        <div className="video-wrapper">
          <video 
            ref={videoRef} 
            autoPlay 
            muted 
            className="camera-video"
          />
          <canvas 
            ref={canvasRef} 
            className="overlay-canvas"
          />
        </div>
        {cameraActive && (
          <div className="camera-status">
            <span className="camera-indicator">📹 Camera Active</span>
          </div>
        )}
      </div>

      {registeredFaces.length > 0 && (
        <div className="registered-faces-list">
          <h3>Registered Faces ({registeredFaces.length})</h3>
          <div className="faces-grid">
            {registeredFaces.map(face => (
              <div key={face.id} className="face-item">
                <span className="face-name">{face.name}</span>
                <span className="face-date">
                  {new Date(face.timestamp).toLocaleDateString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default RegistrationPage
