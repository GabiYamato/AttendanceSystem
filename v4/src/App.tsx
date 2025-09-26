import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import './App.css'
import LoginPage from './components/LoginPage'
import AttendanceSystem from './components/AttendanceSystem'

type UserType = 'admin' | 'teacher' | 'student' | null

function App() {
  const [currentUser, setCurrentUser] = useState<UserType>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Load user from localStorage on app start
  useEffect(() => {
    const savedUser = localStorage.getItem('currentUser')
    if (savedUser) {
      setCurrentUser(savedUser as UserType)
    }
    setIsLoading(false)
  }, [])

  const handleLogin = (userType: UserType) => {
    setCurrentUser(userType)
    if (userType) {
      localStorage.setItem('currentUser', userType)
    }
  }

  const handleLogout = () => {
    setCurrentUser(null)
    localStorage.removeItem('currentUser')
  }

  if (isLoading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        background: 'linear-gradient(135deg, #3C467B 0%, #50589C 50%, #636CCB 100%)'
      }}>
        <div style={{ color: 'white', fontSize: '1.5rem' }}>Loading...</div>
      </div>
    )
  }

  return (
    <Router>
      <div className="app">
        <Routes>
          <Route 
            path="/login" 
            element={
              !currentUser ? (
                <LoginPage onLogin={handleLogin} />
              ) : (
                <Navigate to={currentUser === 'admin' ? '/admin' : '/dashboard'} replace />
              )
            } 
          />
          <Route 
            path="/admin" 
            element={
              currentUser === 'admin' ? (
                <AttendanceSystem onLogout={handleLogout} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/dashboard" 
            element={
              currentUser && currentUser !== 'admin' ? (
                <div className="user-dashboard">
                  <div className="header">
                    <h2>Welcome, Teacher/Student!</h2>
                    <button onClick={handleLogout} className="logout-btn">
                      Logout
                    </button>
                  </div>
                  <div className="dashboard-content">
                    <p>Teacher and Student dashboard functionality coming soon...</p>
                  </div>
                </div>
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/" 
            element={
              <Navigate to={
                !currentUser ? '/login' : 
                currentUser === 'admin' ? '/admin' : '/dashboard'
              } replace />
            } 
          />
        </Routes>
      </div>
    </Router>
  )
}

export default App
