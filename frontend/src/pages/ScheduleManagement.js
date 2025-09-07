import React, { useState } from 'react';
import { Card, Button, Alert } from '../components/UI';
import { Calendar, Plus, Brain, Download, Clock } from 'lucide-react';

export function ScheduleManagement() {
  const [constraints, setConstraints] = useState({
    courses: [''],
    faculty: [''],
    rooms: [''],
    time_slots: [''],
    constraints: ''
  });
  const [generatedSchedule, setGeneratedSchedule] = useState(null);
  const [loading, setLoading] = useState(false);
  const [alert, setAlert] = useState(null);

  const handleArrayInputChange = (field, index, value) => {
    const newArray = [...constraints[field]];
    newArray[index] = value;
    setConstraints({
      ...constraints,
      [field]: newArray
    });
  };

  const addArrayItem = (field) => {
    setConstraints({
      ...constraints,
      [field]: [...constraints[field], '']
    });
  };

  const removeArrayItem = (field, index) => {
    const newArray = constraints[field].filter((_, i) => i !== index);
    setConstraints({
      ...constraints,
      [field]: newArray
    });
  };

  const generateSchedule = async () => {
    setLoading(true);
    setAlert(null);

    try {
      // Filter out empty strings
      const filteredConstraints = {
        courses: constraints.courses.filter(item => item.trim() !== ''),
        faculty: constraints.faculty.filter(item => item.trim() !== ''),
        rooms: constraints.rooms.filter(item => item.trim() !== ''),
        time_slots: constraints.time_slots.filter(item => item.trim() !== ''),
        constraints: constraints.constraints
      };

      const response = await fetch('/api/schedules/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(filteredConstraints)
      });

      const data = await response.json();

      if (data.success) {
        setGeneratedSchedule(data.schedule);
        setAlert({
          type: 'success',
          message: 'Schedule generated successfully using AI optimization!'
        });
      } else {
        throw new Error(data.detail || 'Failed to generate schedule');
      }
    } catch (error) {
      setAlert({
        type: 'error',
        message: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const renderArrayInput = (field, label) => (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      {constraints[field].map((item, index) => (
        <div key={index} className="flex items-center space-x-2 mb-2">
          <input
            type="text"
            value={item}
            onChange={(e) => handleArrayInputChange(field, index, e.target.value)}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder={`Enter ${label.toLowerCase().slice(0, -1)}`}
          />
          {constraints[field].length > 1 && (
            <Button
              type="button"
              variant="danger"
              size="small"
              onClick={() => removeArrayItem(field, index)}
            >
              ×
            </Button>
          )}
        </div>
      ))}
      <Button
        type="button"
        variant="secondary"
        size="small"
        onClick={() => addArrayItem(field)}
      >
        <Plus className="h-4 w-4 mr-1" />
        Add {label.slice(0, -1)}
      </Button>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold text-gray-900">AI Schedule Management</h1>
        <p className="text-gray-600 mt-2">
          Generate optimized class schedules using artificial intelligence
        </p>
      </div>

      {/* Alert */}
      {alert && (
        <Alert 
          type={alert.type} 
          onClose={() => setAlert(null)}
        >
          {alert.message}
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Schedule Generator */}
        <Card>
          <h3 className="text-lg font-medium text-gray-900 flex items-center mb-4">
            <Brain className="h-5 w-5 mr-2" />
            AI Schedule Generator
          </h3>

          <div className="space-y-6">
            {renderArrayInput('courses', 'Courses')}
            {renderArrayInput('faculty', 'Faculty')}
            {renderArrayInput('rooms', 'Rooms')}
            {renderArrayInput('time_slots', 'Time Slots')}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Additional Constraints
              </label>
              <textarea
                value={constraints.constraints}
                onChange={(e) => setConstraints({
                  ...constraints,
                  constraints: e.target.value
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={3}
                placeholder="Enter any additional constraints or preferences..."
              />
            </div>

            <Button 
              onClick={generateSchedule} 
              loading={loading}
              className="w-full"
            >
              <Brain className="h-4 w-4 mr-2" />
              Generate AI Schedule
            </Button>
          </div>
        </Card>

        {/* Generated Schedule Display */}
        <Card>
          <h3 className="text-lg font-medium text-gray-900 flex items-center mb-4">
            <Calendar className="h-5 w-5 mr-2" />
            Generated Schedule
          </h3>

          {generatedSchedule ? (
            <div className="space-y-4">
              {Object.entries(generatedSchedule).map(([day, slots]) => (
                <div key={day} className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">{day}</h4>
                  {slots.length > 0 ? (
                    <div className="space-y-2">
                      {slots.map((slot, index) => (
                        <div key={index} className="bg-blue-50 border border-blue-200 rounded p-3">
                          <div className="flex justify-between items-start">
                            <div>
                              <p className="font-medium text-blue-900">{slot.course}</p>
                              <p className="text-sm text-blue-700">{slot.faculty}</p>
                              <p className="text-sm text-blue-600">{slot.room}</p>
                            </div>
                            <div className="text-right">
                              <p className="text-sm font-medium text-blue-900 flex items-center">
                                <Clock className="h-3 w-3 mr-1" />
                                {slot.time}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-sm">No classes scheduled</p>
                  )}
                </div>
              ))}
              
              <Button variant="secondary" className="w-full">
                <Download className="h-4 w-4 mr-2" />
                Export Schedule
              </Button>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Calendar className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>Generate a schedule to see the results here</p>
            </div>
          )}
        </Card>
      </div>

      {/* Sample Data */}
      <Card className="bg-gray-50 border-gray-200">
        <h4 className="font-medium text-gray-900 mb-3">Sample Data (Click to Use):</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <h5 className="font-medium text-gray-700 mb-2">Courses:</h5>
            <div className="space-y-1">
              {['Data Structures', 'Machine Learning', 'Database Systems', 'Web Development'].map((course) => (
                <button
                  key={course}
                  onClick={() => setConstraints({
                    ...constraints,
                    courses: ['Data Structures', 'Machine Learning', 'Database Systems', 'Web Development']
                  })}
                  className="block text-blue-600 hover:text-blue-800"
                >
                  {course}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <h5 className="font-medium text-gray-700 mb-2">Faculty:</h5>
            <div className="space-y-1">
              {['Dr. Kumar', 'Prof. Rao', 'Dr. Sharma', 'Prof. Gupta'].map((faculty) => (
                <button
                  key={faculty}
                  onClick={() => setConstraints({
                    ...constraints,
                    faculty: ['Dr. Kumar', 'Prof. Rao', 'Dr. Sharma', 'Prof. Gupta']
                  })}
                  className="block text-blue-600 hover:text-blue-800"
                >
                  {faculty}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <h5 className="font-medium text-gray-700 mb-2">Rooms:</h5>
            <div className="space-y-1">
              {['CS-101', 'CS-102', 'Lab-A', 'Lab-B'].map((room) => (
                <button
                  key={room}
                  onClick={() => setConstraints({
                    ...constraints,
                    rooms: ['CS-101', 'CS-102', 'Lab-A', 'Lab-B']
                  })}
                  className="block text-blue-600 hover:text-blue-800"
                >
                  {room}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <h5 className="font-medium text-gray-700 mb-2">Time Slots:</h5>
            <div className="space-y-1">
              {['09:00-10:00', '10:15-11:15', '11:30-12:30', '14:00-15:00'].map((slot) => (
                <button
                  key={slot}
                  onClick={() => setConstraints({
                    ...constraints,
                    time_slots: ['09:00-10:00', '10:15-11:15', '11:30-12:30', '14:00-15:00']
                  })}
                  className="block text-blue-600 hover:text-blue-800"
                >
                  {slot}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
