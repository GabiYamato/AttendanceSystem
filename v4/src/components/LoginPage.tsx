import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './LoginPage.css'

type UserType = 'admin' | 'teacher' | 'student'

interface LoginPageProps {
  onLogin: (userType: UserType) => void
}

const LoginPage: React.FC<LoginPageProps> = ({ onLogin }) => {
  const navigate = useNavigate()
  const [selectedRole, setSelectedRole] = useState<UserType | null>(null)
  const [credentials, setCredentials] = useState({ username: '', password: '' })
  const [error, setError] = useState('')

  const handleRoleSelect = (role: UserType) => {
    setSelectedRole(role)
    setCredentials({ username: '', password: '' })
    setError('')
  }

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (selectedRole === 'admin') {
      if (credentials.username === 'ADMIN' && credentials.password === 'PASSWORD') {
        onLogin('admin')
        navigate('/admin')
      } else {
        setError('Invalid admin credentials')
      }
    } else if (selectedRole === 'teacher' || selectedRole === 'student') {
      if (credentials.username && credentials.password) {
        onLogin(selectedRole)
        navigate('/dashboard')
      } else {
        setError('Please enter username and password')
      }
    }
  }

  const handleBack = () => {
    setSelectedRole(null)
    setCredentials({ username: '', password: '' })
    setError('')
  }

  if (!selectedRole) {
    return (
      <div className="login-container">
        <div className="login-box">
          <h1 className="app-title">Attendance System</h1>
          <p className="app-subtitle">Select your role to continue</p>
          
          <div className="role-buttons">
            <button 
              className="role-btn admin-btn"
              onClick={() => handleRoleSelect('admin')}
            >
              <span>Admin</span>
            </button>
            
            <button 
              className="role-btn teacher-student-btn"
              onClick={() => handleRoleSelect('teacher')}
            >
              <span>Teacher / Student</span>
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="login-container">
      <div className="login-box">
        <button className="back-btn" onClick={handleBack}>
          ← Back
        </button>
        
        <h2 className="login-title">
          {selectedRole === 'admin' ? 'Admin Login' : 'Teacher / Student Login'}
        </h2>
        
        <form onSubmit={handleLogin} className="login-form">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              value={credentials.username}
              onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
              placeholder={selectedRole === 'admin' ? 'Enter ADMIN' : 'Enter your username'}
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={credentials.password}
              onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
              placeholder={selectedRole === 'admin' ? 'Enter PASSWORD' : 'Enter your password'}
              required
            />
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <button type="submit" className="login-submit-btn">
            Login
          </button>
        </form>
      </div>
    </div>
  )
}

export default LoginPage
