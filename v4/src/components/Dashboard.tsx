import React from 'react'
import type { RegisteredFace, AttendanceRecord } from '../types'

interface DashboardProps {
  registeredFaces: RegisteredFace[]
  attendanceRecords: AttendanceRecord[]
  onNavigate: (view: 'register' | 'scan') => void
  onClearData: () => void
}

const Dashboard: React.FC<DashboardProps> = ({
  registeredFaces,
  attendanceRecords,
  onNavigate,
  onClearData
}) => {
  return (
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
          onClick={() => onNavigate('register')}
        >
          📝 Register New Face
        </button>
        <button 
          className="action-btn scan-btn"
          onClick={() => onNavigate('scan')}
        >
          📹 Live Attendance Scan
        </button>
        <button 
          className="action-btn clear-btn"
          onClick={onClearData}
        >
          🗑️ Clear All Data
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
          {attendanceRecords.length === 0 && (
            <div className="no-records">No attendance records yet</div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
