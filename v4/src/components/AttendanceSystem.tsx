import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './AttendanceSystem.css'
import Dashboard from './Dashboard'
import RegistrationPage from './RegistrationPage'
import LiveAttendance from './LiveAttendance'
import {
  getRegisteredFaces,
  getAttendanceRecords,
  clearAllFirebaseData
} from '../services/firebaseService'
import type { RegisteredFace, AttendanceRecord } from '../types'

interface AttendanceSystemProps {
  onLogout: () => void
}

const AttendanceSystem: React.FC<AttendanceSystemProps> = ({ onLogout }) => {
  const navigate = useNavigate()
  const [currentView, setCurrentView] = useState<'dashboard' | 'register' | 'scan'>('dashboard')
  const [registeredFaces, setRegisteredFaces] = useState<RegisteredFace[]>([])
  const [attendanceRecords, setAttendanceRecords] = useState<AttendanceRecord[]>([])

  useEffect(() => {
    loadStoredData()
  }, [])

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

  const handleFaceRegistered = async (newFace: RegisteredFace) => {
    const updatedFaces = [...registeredFaces, newFace]
    setRegisteredFaces(updatedFaces)
    await saveToStorage(updatedFaces, attendanceRecords)
  }

  const handleAttendanceRecorded = async (newRecord: AttendanceRecord) => {
    const updatedRecords = [...attendanceRecords, newRecord]
    setAttendanceRecords(updatedRecords)
    await saveToStorage(registeredFaces, updatedRecords)
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
    setCurrentView(newView)
  }

  const renderNavigation = () => (
    <div className="navigation">
      <div className="nav-left">
        <h1 className="app-title">Attendance System</h1>
      </div>
      <div className="nav-center">
        <button 
          className={`nav-btn ${currentView === 'dashboard' ? 'active' : ''}`}
          onClick={() => handleViewChange('dashboard')}
        >
          📊 Dashboard
        </button>
        <button 
          className={`nav-btn ${currentView === 'register' ? 'active' : ''}`}
          onClick={() => handleViewChange('register')}
        >
          📝 Register
        </button>
        <button 
          className={`nav-btn ${currentView === 'scan' ? 'active' : ''}`}
          onClick={() => handleViewChange('scan')}
        >
          📹 Live Scan
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

  const renderCurrentView = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <Dashboard
            registeredFaces={registeredFaces}
            attendanceRecords={attendanceRecords}
            onNavigate={handleViewChange}
            onClearData={clearAllData}
          />
        )
      case 'register':
        return (
          <RegistrationPage
            registeredFaces={registeredFaces}
            onFaceRegistered={handleFaceRegistered}
            onBack={() => handleViewChange('dashboard')}
          />
        )
      case 'scan':
        return (
          <LiveAttendance
            registeredFaces={registeredFaces}
            attendanceRecords={attendanceRecords}
            onAttendanceRecorded={handleAttendanceRecorded}
            onBack={() => handleViewChange('dashboard')}
          />
        )
      default:
        return null
    }
  }

  return (
    <div className="attendance-system">
      {renderNavigation()}
      <div className="main-content">
        {renderCurrentView()}
      </div>
    </div>
  )
}

export default AttendanceSystem
