import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

interface Class {
  class_id: string;
  students_count: number;
}

const Dashboard: React.FC = () => {
  const [classes, setClasses] = useState<Class[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/classes');
      setClasses(response.data.classes);
    } catch (error) {
      console.error('Error fetching classes:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center">Loading classes...</div>;
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-8">Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {classes.map((classItem) => (
          <div key={classItem.class_id} className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-semibold mb-2">Class: {classItem.class_id}</h3>
            <p className="text-gray-600 mb-4">Students: {classItem.students_count}</p>
            <div className="space-x-2">
              <Link 
                to={`/class/${classItem.class_id}`}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
              >
                Manage
              </Link>
              <Link 
                to={`/attendance/${classItem.class_id}`}
                className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
              >
                Attendance
              </Link>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8 text-center">
        <Link 
          to="/register"
          className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700"
        >
          Register New Student
        </Link>
      </div>
    </div>
  );
};

export default Dashboard;
