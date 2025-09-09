import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';

interface Student {
  student_id: string;
  name: string;
  face_registered: boolean;
  registered_at: string | null;
}

const ClassDashboard: React.FC = () => {
  const { classId } = useParams<{ classId: string }>();
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (classId) {
      fetchStudents();
    }
  }, [classId]);

  const fetchStudents = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/api/classes/${classId}/students`);
      setStudents(response.data.students);
    } catch (error) {
      console.error('Error fetching students:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center">Loading students...</div>;
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-8">Class {classId} - Students</h1>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="mb-4 flex justify-between items-center">
          <h2 className="text-xl font-semibold">Students ({students.length})</h2>
        </div>

        {students.length === 0 ? (
          <p className="text-gray-500">No students registered in this class yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full table-auto">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-4 py-2 text-left">Student ID</th>
                  <th className="px-4 py-2 text-left">Name</th>
                  <th className="px-4 py-2 text-left">Face Registered</th>
                  <th className="px-4 py-2 text-left">Registered At</th>
                </tr>
              </thead>
              <tbody>
                {students.map((student) => (
                  <tr key={student.student_id} className="border-t">
                    <td className="px-4 py-2">{student.student_id}</td>
                    <td className="px-4 py-2 font-medium">{student.name}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 rounded text-xs ${
                        student.face_registered 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {student.face_registered ? 'Yes' : 'No'}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-sm text-gray-600">
                      {student.registered_at 
                        ? new Date(student.registered_at).toLocaleDateString()
                        : 'N/A'
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default ClassDashboard;
