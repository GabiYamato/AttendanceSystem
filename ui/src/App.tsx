import { useState } from 'react';
import { StudentRegistrationForm } from './components/StudentRegistrationForm';
import { AttendanceMarking } from './components/AttendanceMarking';

type View = 'home' | 'register' | 'attendance';

function App() {
  const [currentView, setCurrentView] = useState<View>('home');

  const renderView = () => {
    switch (currentView) {
      case 'register':
        return <StudentRegistrationForm onSuccess={() => setCurrentView('home')} />;
      case 'attendance':
        return <AttendanceMarking />;
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-100 flex items-center justify-center p-6">
            <div className="max-w-4xl mx-auto text-center">
              {/* Hero Section */}
              <div className="mb-12">
                <h1 className="text-6xl font-bold text-gray-800 mb-4">
                  Face Recognition
                  <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent block">
                    Attendance System
                  </span>
                </h1>
                <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                  Advanced facial recognition technology for seamless student attendance tracking. 
                  Secure, fast, and accurate.
                </p>
              </div>

              {/* Action Cards */}
              <div className="grid md:grid-cols-2 gap-8 mb-12">
                <div 
                  onClick={() => setCurrentView('register')}
                  className="bg-white rounded-2xl shadow-xl p-8 cursor-pointer transform hover:scale-105 transition-all duration-200 hover:shadow-2xl group"
                >
                  <div className="text-6xl mb-4 group-hover:scale-110 transition-transform">👤</div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-3">Student Registration</h3>
                  <p className="text-gray-600 mb-6">
                    Register as a new student and capture your face for future recognition
                  </p>
                  <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold">
                    Get Started →
                  </div>
                </div>

                <div 
                  onClick={() => setCurrentView('attendance')}
                  className="bg-white rounded-2xl shadow-xl p-8 cursor-pointer transform hover:scale-105 transition-all duration-200 hover:shadow-2xl group"
                >
                  <div className="text-6xl mb-4 group-hover:scale-110 transition-transform">📋</div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-3">Mark Attendance</h3>
                  <p className="text-gray-600 mb-6">
                    Use face recognition to quickly mark your attendance for classes
                  </p>
                  <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white py-3 px-6 rounded-lg font-semibold">
                    Scan Face →
                  </div>
                </div>
              </div>

              {/* Features */}
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <h3 className="text-2xl font-bold text-gray-800 mb-6">Key Features</h3>
                <div className="grid md:grid-cols-3 gap-6 text-left">
                  <div className="flex items-start space-x-3">
                    <div className="text-2xl">⚡</div>
                    <div>
                      <h4 className="font-semibold text-gray-800">Fast Recognition</h4>
                      <p className="text-gray-600 text-sm">Instant face detection and recognition</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="text-2xl">🔒</div>
                    <div>
                      <h4 className="font-semibold text-gray-800">Secure & Private</h4>
                      <p className="text-gray-600 text-sm">Advanced encryption and data protection</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="text-2xl">📊</div>
                    <div>
                      <h4 className="font-semibold text-gray-800">Real-time Analytics</h4>
                      <p className="text-gray-600 text-sm">Live attendance tracking and reports</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-100">
      {/* Navigation */}
      {currentView !== 'home' && (
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div 
                onClick={() => setCurrentView('home')}
                className="flex items-center cursor-pointer"
              >
                <span className="text-xl font-bold text-gray-800">Face Recognition System</span>
              </div>
              <div className="flex space-x-4">
                <button
                  onClick={() => setCurrentView('register')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'register' 
                      ? 'bg-blue-600 text-white' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  Register
                </button>
                <button
                  onClick={() => setCurrentView('attendance')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'attendance' 
                      ? 'bg-green-600 text-white' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  Attendance
                </button>
              </div>
            </div>
          </div>
        </nav>
      )}

      {/* Main Content */}
      <main className={currentView === 'home' ? '' : 'py-8'}>
        {renderView()}
      </main>
    </div>
  );
}

export default App;
