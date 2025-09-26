import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import * as faceapi from 'face-api.js'
import './AttendanceSystem.css'
import {
  saveRegisteredFace,
  getRegisteredFaces,
  saveAttendanceRecord,
  getAttendanceRecords,
  clearAllFirebaseData
} from '../services/firebaseService'
import type { 
  RegisteredFace as FirebaseRegisteredFace,
  AttendanceRecord as FirebaseAttendanceRecord
} from '../services/firebaseService'

interface AttendanceSystemProps {
  onLogout: () => void
}

interface RegisteredFace {
  id: string
  name: string
  descriptor: Float32Array
  timestamp: string
}

interface AttendanceRecord {
  id: string
  name: string
  timestamp: string
  status: 'present' | 'absent'
}

const AttendanceSystem: React.FC<AttendanceSystemProps> = ({ onLogout }) => {
  const navigate = useNavigate()
  const [currentView, setCurrentView] = useState<'dashboard' | 'register' | 'scan'>('dashboard')
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [registeredFaces, setRegisteredFaces] = useState<RegisteredFace[]>([])
  const [attendanceRecords, setAttendanceRecords] = useState<AttendanceRecord[]>([])
  const [isScanning, setIsScanning] = useState(false)
  const [scanResults, setScanResults] = useState<string[]>([])
  
  // Registration states
  const [registerName, setRegisterName] = useState('')
  const [registerStatus, setRegisterStatus] = useState('')
  
  // Camera state
  const [cameraActive, setCameraActive] = useState(false)
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    loadModels()
    loadStoredData()
  }, [])

  const loadModels = async () => {
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models')
      ])
      setIsModelLoaded(true)
      console.log('Face API models loaded successfully')
    } catch (error) {
      console.error('Error loading models:', error)
      setRegisterStatus('Error loading face recognition models')
    }
  }

  const loadStoredData = async () => {
    try {
      // Load from Firebase
      const [firebaseFaces, firebaseRecords] = await Promise.all([
        getRegisteredFaces(),
        getAttendanceRecords()
      ])
      
      // Convert Firebase data to local format
      const faces = firebaseFaces.map(face => {
        const convertedFace = {
          ...face,
          descriptor: new Float32Array(face.descriptor)
        }
        console.log(`Loaded face: ${face.name}, descriptor length: ${face.descriptor.length}`)
        return convertedFace
      })
      
      console.log('Loaded registered faces from Firebase:', faces.map(f => f.name))
      console.log('Total faces loaded:', faces.length)
      console.log('Loaded attendance records from Firebase:', firebaseRecords.length, 'records')
      
      setRegisteredFaces(faces)
      setAttendanceRecords(firebaseRecords)
    } catch (error) {
      console.error('Error loading data from Firebase:', error)
      // Fallback to localStorage if Firebase fails
      const storedFaces = localStorage.getItem('registeredFaces')
      const storedRecords = localStorage.getItem('attendanceRecords')
      
      if (storedFaces) {
        const faces = JSON.parse(storedFaces).map((face: any) => ({
          ...face,
          descriptor: new Float32Array(face.descriptor)
        }))
        console.log('Loaded faces from localStorage:', faces.length)
        setRegisteredFaces(faces)
      }
      
      if (storedRecords) {
        setAttendanceRecords(JSON.parse(storedRecords))
      }
    }
  }

  const saveToStorage = async (faces: RegisteredFace[], records: AttendanceRecord[]) => {
    console.log('Saving to storage:', faces.length, 'faces and', records.length, 'records')
    try {
      // Save to localStorage as backup
      localStorage.setItem('registeredFaces', JSON.stringify(faces))
      localStorage.setItem('attendanceRecords', JSON.stringify(records))
      console.log('Data saved to localStorage as backup')
    } catch (error) {
      console.error('Error saving to localStorage:', error)
    }
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setCameraActive(true)
      }
      return true
    } catch (error) {
      console.error('Error accessing camera:', error)
      setRegisterStatus('Error accessing camera')
      return false
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
      setCameraActive(false)
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      }
    }
  }

  const handleRegisterFace = async () => {
    if (!registerName.trim()) {
      setRegisterStatus('Please enter a name')
      return
    }

    if (!isModelLoaded) {
      setRegisterStatus('Models are still loading...')
      return
    }

    const cameraStarted = await startCamera()
    if (!cameraStarted) return

    setRegisterStatus('Position your face in the camera and wait...')
    console.log('Starting face registration for:', registerName)

    // Start continuous detection for registration
    const startRegistrationDetection = async () => {
      if (!videoRef.current || !canvasRef.current) return

      const canvas = canvasRef.current
      const video = videoRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Wait for video to be ready and set canvas size
      const waitForVideo = () => {
        return new Promise<void>((resolve) => {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            console.log(`Registration canvas set to ${canvas.width}x${canvas.height}`)
            resolve()
          } else {
            const handleLoadedMetadata = () => {
              canvas.width = video.videoWidth
              canvas.height = video.videoHeight
              console.log(`Registration canvas set to ${canvas.width}x${canvas.height}`)
              video.removeEventListener('loadedmetadata', handleLoadedMetadata)
              resolve()
            }
            video.addEventListener('loadedmetadata', handleLoadedMetadata)
          }
        })
      }

      await waitForVideo()

      let detectionCount = 0
      const maxDetections = 5 // Reduced for faster registration
      let descriptors: Float32Array[] = []

      const detectForRegistration = async () => {
        if (detectionCount >= maxDetections) return

        try {
          const detection = await faceapi
            .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 }))
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
            ctx.fillRect(box.x, box.y - 35, box.width, 120)
            
            // Draw label
            ctx.fillStyle = '#ffffff'
            ctx.font = 'bold 16px Arial'
            ctx.textAlign = 'center'
            ctx.fillText(
              `Registering: ${registerName}`,
              box.x + box.width / 2,
              box.y - 10
            )
            ctx.textAlign = 'start' // Reset text alignment

            detectionCount++
            descriptors.push(detection.descriptor)
            console.log(`Face detected for registration (${detectionCount}/${maxDetections})`)

            if (detectionCount >= maxDetections) {
              // Average the descriptors for better accuracy
              const avgDescriptor = new Float32Array(descriptors[0].length)
              for (let i = 0; i < avgDescriptor.length; i++) {
                let sum = 0
                for (const desc of descriptors) {
                  sum += desc[i]
                }
                avgDescriptor[i] = sum / descriptors.length
              }

              // Register the face
              const newFace: RegisteredFace = {
                id: `face-${Date.now()}`,
                name: registerName.trim(),
                descriptor: avgDescriptor,
                timestamp: new Date().toISOString()
              }

              try {
                // Save to Firebase
                await saveRegisteredFace({
                  ...newFace,
                  descriptor: Array.from(avgDescriptor) // Convert Float32Array to regular array
                })
                
                const updatedFaces = [...registeredFaces, newFace]
                setRegisteredFaces(updatedFaces)
                await saveToStorage(updatedFaces, attendanceRecords)
                setRegisterStatus(`Successfully registered ${registerName}!`)
                console.log('Face successfully registered:', newFace.name, 'with averaged descriptor')
                setRegisterName('')
              } catch (error) {
                console.error('Error saving face:', error)
                setRegisterStatus('Error saving face. Please try again.')
              }
              
              setTimeout(() => {
                stopCamera()
                setCurrentView('dashboard')
                setRegisterStatus('')
              }, 2000)
              return
            }
          } else {
            console.log('No face detected during registration')
          }
        } catch (error) {
          console.error('Error in registration detection:', error)
        }

        setTimeout(detectForRegistration, 200) // Slightly slower for registration
      }

      // Start detection after a short delay
      setTimeout(detectForRegistration, 1000)
    }

    setTimeout(startRegistrationDetection, 1000)
  }

  const handleStartScanning = async () => {
    if (!isModelLoaded) {
      alert('Models are still loading...')
      return
    }

    const cameraStarted = await startCamera()
    if (!cameraStarted) return

    setIsScanning(true)
    setScanResults([])
    scanForFaces()
  }

  const scanForFaces = async () => {
    if (!videoRef.current || !canvasRef.current) return

    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    console.log('Starting face scanning with', registeredFaces.length, 'registered faces')

    // Wait for video to be ready and set canvas size
    const waitForVideo = () => {
      return new Promise<void>((resolve) => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          console.log(`Canvas set to ${canvas.width}x${canvas.height}`)
          resolve()
        } else {
          const handleLoadedMetadata = () => {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            console.log(`Canvas set to ${canvas.width}x${canvas.height}`)
            video.removeEventListener('loadedmetadata', handleLoadedMetadata)
            resolve()
          }
          video.addEventListener('loadedmetadata', handleLoadedMetadata)
        }
      })
    }

    await waitForVideo()

    const detectFaces = async () => {
      if (!isScanning || !videoRef.current || !canvasRef.current) return

      try {
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.5 }))
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
              0.5 // Lower threshold for better matching
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
                ctx.fillRect(box.x, box.y - 35, box.width, 35)
                
                // Draw label
                ctx.fillStyle = '#ffffff'
                ctx.font = 'bold 16px Arial'
                ctx.textAlign = 'center'
                ctx.fillText(
                  `${match.label} (${confidence}%)`,
                  box.x + box.width / 2,
                  box.y - 10
                )
                ctx.textAlign = 'start' // Reset text alignment

                // Record attendance only once per session per person
                if (!scanResults.includes(match.label)) {
                  console.log('Recording attendance for:', match.label)
                  setScanResults(prev => [...prev, match.label])
                  
                  const newRecord: AttendanceRecord = {
                    id: `${Date.now()}-${match.label}`,
                    name: match.label,
                    timestamp: new Date().toISOString(),
                    status: 'present'
                  }
                  
                  try {
                    // Save to Firebase
                    await saveAttendanceRecord(newRecord)
                    
                    const updatedRecords = [...attendanceRecords, newRecord]
                    setAttendanceRecords(updatedRecords)
                    await saveToStorage(registeredFaces, updatedRecords)
                    console.log('Attendance recorded successfully for:', match.label)
                  } catch (error) {
                    console.error('Error recording attendance:', error)
                  }
                }
              } else {
                // Unknown face - red box
                ctx.strokeStyle = '#e74c3c'
                ctx.lineWidth = 3
                ctx.strokeRect(box.x, box.y, box.width, box.height)
                
                // Draw background for text
                ctx.fillStyle = 'rgba(231, 76, 60, 0.8)'
                ctx.fillRect(box.x, box.y - 35, box.width, 35)
                
                ctx.fillStyle = '#ffffff'
                ctx.font = 'bold 16px Arial'
                ctx.textAlign = 'center'
                ctx.fillText('Unknown', box.x + box.width / 2, box.y - 10)
                ctx.textAlign = 'start' // Reset text alignment
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
              ctx.fillRect(box.x, box.y - 35, box.width, 35)
              
              ctx.fillStyle = '#ffffff'
              ctx.font = 'bold 16px Arial'
              ctx.textAlign = 'center'
              ctx.fillText('Unknown', box.x + box.width / 2, box.y - 10)
              ctx.textAlign = 'start' // Reset text alignment
            }
          }
        }
      } catch (error) {
        console.error('Error in face detection:', error)
      }

      // Continue scanning with proper timing
      if (isScanning) {
        requestAnimationFrame(detectFaces)
      }
    }

    // Start detection loop
    detectFaces()
  }

  const handleStopScanning = () => {
    setIsScanning(false)
    stopCamera()
  }

  const clearAllData = async () => {
    if (confirm('Are you sure you want to clear all registered faces and attendance records?')) {
      try {
        // Clear from Firebase
        await clearAllFirebaseData()
        
        // Clear local state
        setRegisteredFaces([])
        setAttendanceRecords([])
        
        // Clear localStorage backup
        localStorage.removeItem('registeredFaces')
        localStorage.removeItem('attendanceRecords')
        
        console.log('All data cleared successfully')
      } catch (error) {
        console.error('Error clearing data:', error)
        alert('Error clearing data. Please try again.')
      }
    }
  }

  const handleViewChange = (newView: 'dashboard' | 'register' | 'scan') => {
    // Stop camera and scanning when changing views
    setIsScanning(false)
    stopCamera()
    setRegisterStatus('')
    setScanResults([])
    setCurrentView(newView)
  }

  const renderNavigation = () => (
    <div className="navigation">
      <div className="nav-left">
        <h1 className="app-title">Attendance System</h1>
        {cameraActive && <span className="camera-indicator">📹 Camera Active</span>}
      </div>
      <div className="nav-center">
        <button 
          className={`nav-btn ${currentView === 'dashboard' ? 'active' : ''}`}
          onClick={() => handleViewChange('dashboard')}
        >
           Dashboard
        </button>
        <button 
          className={`nav-btn ${currentView === 'register' ? 'active' : ''}`}
          onClick={() => handleViewChange('register')}
        >
           Register
        </button>
        <button 
          className={`nav-btn ${currentView === 'scan' ? 'active' : ''}`}
          onClick={() => handleViewChange('scan')}
        >
           Live Scan
        </button>
      </div>
      <div className="nav-right">
        <button onClick={() => {
          onLogout()
          navigate('/login')
        }} className="logout-btn">Logout</button>
      </div>
    </div>
  )

  const renderDashboard = () => (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Admin Dashboard</h2>
        <div className="stats-summary">
          Registered: {registeredFaces.length} | Records: {attendanceRecords.length}
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">{registeredFaces.length}</div>
          <div className="stat-label">Registered Faces</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{attendanceRecords.length}</div>
          <div className="stat-label">Attendance Records</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">
            {new Set(attendanceRecords.map(r => r.name)).size}
          </div>
          <div className="stat-label">Unique Attendees</div>
        </div>
      </div>

      <div className="action-buttons">
        <button 
          className="action-btn register-btn"
          onClick={() => handleViewChange('register')}
        >
           Register New Face
        </button>
        <button 
          className="action-btn scan-btn"
          onClick={() => handleViewChange('scan')}
        >
           Live Attendance Scan
        </button>
        <button 
          className="action-btn clear-btn"
          onClick={clearAllData}
        >
           Clear All Data
        </button>
      </div>

      <div className="recent-records">
        <h3>Recent Attendance Records</h3>
        <div className="records-list">
          {attendanceRecords.slice(-10).reverse().map(record => (
            <div key={record.id} className="record-item">
              <span className="record-name">{record.name}</span>
              <span className="record-time">
                {new Date(record.timestamp).toLocaleString()}
              </span>
              <span className={`record-status ${record.status}`}>
                {record.status}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderRegister = () => (
    <div className="register-view">
      <div className="view-header">
        <h2>Register New Face</h2>
        <p className="view-description">Enter a name and position the face in the camera to register</p>
      </div>

      <div className="register-form">
        <div className="form-group">
          <label>Name:</label>
          <input
            type="text"
            value={registerName}
            onChange={(e) => setRegisterName(e.target.value)}
            placeholder="Enter person's name"
          />
        </div>
        
        <button 
          onClick={handleRegisterFace}
          className="register-face-btn"
          disabled={!registerName.trim() || !isModelLoaded}
        >
          {!isModelLoaded ? 'Loading Models...' : 'Start Registration'}
        </button>
        
        {registerStatus && (
          <div className="status-message">{registerStatus}</div>
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
      </div>
    </div>
  )

  const renderScan = () => (
    <div className="scan-view">
      <div className="view-header">
        <h2>Live Attendance Scan</h2>
        <p className="view-description">Start scanning to detect and mark attendance for registered faces</p>
      </div>

      <div className="scan-controls">
        {!isScanning ? (
          <button onClick={handleStartScanning} className="start-scan-btn">
             Start Scanning
          </button>
        ) : (
          <div className="scanning-active">
            <button onClick={handleStopScanning} className="stop-scan-btn">
               Stop Scanning
            </button>
            <div className="scanning-indicator">
              <span className="pulse-dot"></span>
              Live scanning in progress...
            </div>
          </div>
        )}
      </div>

      {(scanResults.length > 0 || isScanning) && (
        <div className="scan-results">
          <h3>Recognition Results</h3>
          {scanResults.length > 0 ? (
            <div>
              <p className="results-summary"> Attendance marked for {scanResults.length} person(s):</p>
              {scanResults.map((name, index) => (
                <span key={index} className="detected-name">{name}</span>
              ))}
            </div>
          ) : isScanning ? (
            <p className="waiting-message"> Scanning for faces...</p>
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
      </div>
    </div>
  )

  return (
    <div className="attendance-system">
      {renderNavigation()}
      <div className="main-content">
        {currentView === 'dashboard' && renderDashboard()}
        {currentView === 'register' && renderRegister()}
        {currentView === 'scan' && renderScan()}
      </div>
    </div>
  )
}

export default AttendanceSystem
