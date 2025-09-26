import React, { useState, useCallback, useRef } from 'react'
import * as faceapi from 'face-api.js'
import { useFaceAPI } from '../hooks/useFaceAPI'
import { useCamera } from '../hooks/useCamera'
import type { RegisteredFace, AttendanceRecord } from '../types'
import { saveAttendanceRecord } from '../services/firebaseService'

interface LiveAttendanceProps {
  registeredFaces: RegisteredFace[]
  attendanceRecords: AttendanceRecord[]
  onAttendanceRecorded: (record: AttendanceRecord) => void
  onBack: () => void
}

const LiveAttendance: React.FC<LiveAttendanceProps> = ({
  registeredFaces,
  attendanceRecords,
  onAttendanceRecorded,
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
  
  const [isScanning, setIsScanning] = useState(false)
  const [scanResults, setScanResults] = useState<string[]>([])
  const [sessionAttendance, setSessionAttendance] = useState<string[]>([])
  const scanningRef = useRef<boolean>(false)

  const handleStartScanning = useCallback(async () => {
    if (!isModelLoaded) {
      alert('Models are still loading...')
      return
    }

    console.log('Starting live attendance scanning...')
    const cameraStarted = await startCamera()
    if (!cameraStarted) return

    setIsScanning(true)
    scanningRef.current = true
    setScanResults([])
    setSessionAttendance([])
    
    // Start scanning after camera is ready
    setTimeout(startFaceDetection, 1000)
  }, [isModelLoaded, startCamera])

  const startFaceDetection = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    console.log('Setting up canvas for live detection...')
    await setupCanvas()
    console.log('Starting face detection with', registeredFaces.length, 'registered faces')

    const detectFaces = async () => {
      if (!scanningRef.current || !videoRef.current || !canvasRef.current) return

      try {
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ 
            inputSize: 416, 
            scoreThreshold: 0.5 
          }))
          .withFaceLandmarks()
          .withFaceDescriptors()

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        console.log(`Detected ${detections.length} faces in current frame`)

        if (detections.length > 0) {
          if (registeredFaces.length > 0) {
            // Create face matcher with registered faces
            const faceMatcher = new faceapi.FaceMatcher(
              registeredFaces.map(face => 
                new faceapi.LabeledFaceDescriptors(face.name, [face.descriptor])
              ),
              0.5 // Distance threshold for matching
            )

            for (let i = 0; i < detections.length; i++) {
              const detection = detections[i]
              const match = faceMatcher.findBestMatch(detection.descriptor)
              const box = detection.detection.box
              const confidence = Math.round((1 - match.distance) * 100)
              
              console.log(`Face ${i + 1}: ${match.label} (confidence: ${confidence}%, distance: ${match.distance.toFixed(3)})`)
              
              if (match.label !== 'unknown' && match.distance < 0.5) {
                // Recognized face - green box
                ctx.strokeStyle = '#27ae60'
                ctx.lineWidth = 3
                ctx.strokeRect(box.x, box.y, box.width, box.height)
                
                // Draw background for text
                ctx.fillStyle = 'rgba(39, 174, 96, 0.8)'
                ctx.fillRect(box.x, box.y - 35, Math.max(box.width, 200), 35)
                
                // Draw label
                ctx.fillStyle = '#ffffff'
                ctx.font = 'bold 16px Arial'
                ctx.textAlign = 'center'
                ctx.fillText(
                  `${match.label} (${confidence}%)`,
                  box.x + box.width / 2,
                  box.y - 10
                )
                ctx.textAlign = 'start'

                // Record attendance only once per session per person
                if (!sessionAttendance.includes(match.label)) {
                  console.log('Recording attendance for:', match.label)
                  await recordAttendance(match.label)
                }
              } else {
                // Unknown face - red box
                ctx.strokeStyle = '#e74c3c'
                ctx.lineWidth = 3
                ctx.strokeRect(box.x, box.y, box.width, box.height)
                
                // Draw background for text
                ctx.fillStyle = 'rgba(231, 76, 60, 0.8)'
                ctx.fillRect(box.x, box.y - 35, Math.max(box.width, 100), 35)
                
                ctx.fillStyle = '#ffffff'
                ctx.font = 'bold 16px Arial'
                ctx.textAlign = 'center'
                ctx.fillText('Unknown', box.x + box.width / 2, box.y - 10)
                ctx.textAlign = 'start'
              }
            }
          } else {
            // No registered faces - show all as unknown
            for (let i = 0; i < detections.length; i++) {
              const detection = detections[i]
              const box = detection.detection.box
              console.log('No registered faces - showing as unknown')
              
              ctx.strokeStyle = '#e74c3c'
              ctx.lineWidth = 3
              ctx.strokeRect(box.x, box.y, box.width, box.height)
              
              // Draw background for text
              ctx.fillStyle = 'rgba(231, 76, 60, 0.8)'
              ctx.fillRect(box.x, box.y - 35, Math.max(box.width, 100), 35)
              
              ctx.fillStyle = '#ffffff'
              ctx.font = 'bold 16px Arial'
              ctx.textAlign = 'center'
              ctx.fillText('Unknown', box.x + box.width / 2, box.y - 10)
              ctx.textAlign = 'start'
            }
          }
        } else {
          // No faces detected - show status
          ctx.fillStyle = 'rgba(52, 152, 219, 0.8)'
          ctx.fillRect(20, 20, canvas.width - 40, 60)
          ctx.fillStyle = '#ffffff'
          ctx.font = 'bold 18px Arial'
          ctx.textAlign = 'center'
          ctx.fillText('Scanning for faces...', canvas.width / 2, 55)
          ctx.textAlign = 'start'
        }
      } catch (error) {
        console.error('Error in face detection:', error)
      }

      // Continue scanning
      if (scanningRef.current) {
        requestAnimationFrame(detectFaces)
      }
    }

    // Start detection loop
    detectFaces()
  }, [registeredFaces, setupCanvas, sessionAttendance])

  const recordAttendance = useCallback(async (name: string) => {
    try {
      const newRecord: AttendanceRecord = {
        id: `${Date.now()}-${name}`,
        name: name,
        timestamp: new Date().toISOString(),
        status: 'present'
      }
      
      // Save to Firebase
      await saveAttendanceRecord(newRecord)
      
      // Update local state
      onAttendanceRecorded(newRecord)
      setScanResults(prev => [...prev, name])
      setSessionAttendance(prev => [...prev, name])
      
      console.log('Attendance recorded successfully for:', name)
    } catch (error) {
      console.error('Error recording attendance:', error)
    }
  }, [onAttendanceRecorded])

  const handleStopScanning = useCallback(() => {
    console.log('Stopping live attendance scanning...')
    setIsScanning(false)
    scanningRef.current = false
    stopCamera()
  }, [stopCamera])

  const handleClearSession = useCallback(() => {
    setScanResults([])
    setSessionAttendance([])
    console.log('Session attendance cleared')
  }, [])

  return (
    <div className="scan-view">
      <div className="view-header">
        <h2>Live Attendance Scan</h2>
        <p className="view-description">
          Start scanning to detect and mark attendance for registered faces
        </p>
        {registeredFaces.length === 0 && (
          <div className="warning-message">
            ⚠️ No registered faces found. Please register faces first.
          </div>
        )}
      </div>

      <div className="scan-controls">
        {!isScanning ? (
          <div className="control-buttons">
            <button 
              onClick={handleStartScanning} 
              className="start-scan-btn"
              disabled={!isModelLoaded}
            >
              {!isModelLoaded ? 'Loading Models...' : '📹 Start Scanning'}
            </button>
            <button 
              onClick={onBack}
              className="back-btn"
            >
              Back to Dashboard
            </button>
          </div>
        ) : (
          <div className="scanning-active">
            <button onClick={handleStopScanning} className="stop-scan-btn">
              ⏹️ Stop Scanning
            </button>
            <button onClick={handleClearSession} className="clear-session-btn">
              🗑️ Clear Session
            </button>
            <div className="scanning-indicator">
              <span className="pulse-dot"></span>
              Live scanning in progress...
            </div>
          </div>
        )}
      </div>

      {(modelError || cameraError) && (
        <div className="error-message">
          {modelError || cameraError}
        </div>
      )}

      {(scanResults.length > 0 || isScanning) && (
        <div className="scan-results">
          <h3>Session Results</h3>
          {scanResults.length > 0 ? (
            <div>
              <p className="results-summary">
                ✅ Attendance marked for {scanResults.length} person(s):
              </p>
              <div className="detected-names">
                {scanResults.map((name, index) => (
                  <span key={index} className="detected-name">{name}</span>
                ))}
              </div>
            </div>
          ) : isScanning ? (
            <p className="waiting-message">🔍 Scanning for faces...</p>
          ) : null}
        </div>
      )}

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
            <span className="camera-indicator">📹 Camera Active - Live Detection</span>
            <span className="registered-count">
              Registered faces: {registeredFaces.length}
            </span>
          </div>
        )}
      </div>

      {/* Statistics */}
      <div className="scan-stats">
        <div className="stat-item">
          <span className="stat-label">Total Registered:</span>
          <span className="stat-value">{registeredFaces.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Session Detected:</span>
          <span className="stat-value">{scanResults.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total Records:</span>
          <span className="stat-value">{attendanceRecords.length}</span>
        </div>
      </div>
    </div>
  )
}

export default LiveAttendance
