import React, { useState, useEffect } from 'react';
import { Card, Button } from '../components/UI';
import { BarChart3, Download, Calendar, Users, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

export function Reports() {
  const [reportData, setReportData] = useState({
    attendance_trends: [],
    class_stats: [],
    monthly_summary: []
  });
  const [loading, setLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState('week');

  useEffect(() => {
    fetchReportData();
  }, [selectedPeriod]);

  const fetchReportData = async () => {
    setLoading(true);
    try {
      // Simulate report data - replace with actual API calls
      const sampleData = {
        attendance_trends: [
          { date: '2024-01-01', attendance: 85 },
          { date: '2024-01-02', attendance: 92 },
          { date: '2024-01-03', attendance: 78 },
          { date: '2024-01-04', attendance: 88 },
          { date: '2024-01-05', attendance: 95 },
          { date: '2024-01-06', attendance: 82 },
          { date: '2024-01-07', attendance: 90 }
        ],
        class_stats: [
          { class: 'CS-A', present: 28, total: 30 },
          { class: 'CS-B', present: 25, total: 28 },
          { class: 'ECE-A', present: 32, total: 35 },
          { class: 'ECE-B', present: 29, total: 32 }
        ],
        monthly_summary: [
          { month: 'Jan', sessions: 45, avg_attendance: 87 },
          { month: 'Feb', sessions: 42, avg_attendance: 89 },
          { month: 'Mar', sessions: 48, avg_attendance: 85 },
          { month: 'Apr', sessions: 46, avg_attendance: 91 }
        ]
      };
      
      setReportData(sampleData);
    } catch (error) {
      console.error('Error fetching report data:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportReport = (type) => {
    // Simulate export functionality
    console.log(`Exporting ${type} report...`);
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="animate-pulse">
              <div className="h-32 bg-gray-200 rounded"></div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Reports & Analytics</h1>
          <p className="text-gray-600">Attendance insights and statistics</p>
        </div>
        <div className="flex space-x-3">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="quarter">This Quarter</option>
          </select>
          <Button variant="secondary">
            <Download className="h-4 w-4 mr-2" />
            Export All
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Sessions</p>
              <p className="text-2xl font-bold text-gray-900">156</p>
              <p className="text-xs text-green-600 mt-1">+12% from last month</p>
            </div>
            <div className="p-3 rounded-lg bg-blue-100 text-blue-600">
              <Calendar className="h-6 w-6" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Attendance</p>
              <p className="text-2xl font-bold text-gray-900">88.5%</p>
              <p className="text-xs text-green-600 mt-1">+3.2% from last month</p>
            </div>
            <div className="p-3 rounded-lg bg-green-100 text-green-600">
              <TrendingUp className="h-6 w-6" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Students</p>
              <p className="text-2xl font-bold text-gray-900">125</p>
              <p className="text-xs text-blue-600 mt-1">5 new this month</p>
            </div>
            <div className="p-3 rounded-lg bg-purple-100 text-purple-600">
              <Users className="h-6 w-6" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Recognition Rate</p>
              <p className="text-2xl font-bold text-gray-900">96.8%</p>
              <p className="text-xs text-green-600 mt-1">+1.5% improvement</p>
            </div>
            <div className="p-3 rounded-lg bg-yellow-100 text-yellow-600">
              <BarChart3 className="h-6 w-6" />
            </div>
          </div>
        </Card>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Attendance Trends */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Attendance Trends</h3>
            <Button variant="secondary" size="small" onClick={() => exportReport('trends')}>
              <Download className="h-3 w-3 mr-1" />
              Export
            </Button>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={reportData.attendance_trends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="attendance" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Class Statistics */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Class Statistics</h3>
            <Button variant="secondary" size="small" onClick={() => exportReport('classes')}>
              <Download className="h-3 w-3 mr-1" />
              Export
            </Button>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={reportData.class_stats}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="present" fill="#22c55e" name="Present" />
              <Bar dataKey="total" fill="#e5e7eb" name="Total" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Detailed Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Summary */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Monthly Summary</h3>
            <Button variant="secondary" size="small" onClick={() => exportReport('monthly')}>
              <Download className="h-3 w-3 mr-1" />
              Export
            </Button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 text-gray-600">Month</th>
                  <th className="text-right py-2 text-gray-600">Sessions</th>
                  <th className="text-right py-2 text-gray-600">Avg Attendance</th>
                </tr>
              </thead>
              <tbody>
                {reportData.monthly_summary.map((month, index) => (
                  <tr key={index} className="border-b border-gray-100">
                    <td className="py-2 font-medium text-gray-900">{month.month}</td>
                    <td className="py-2 text-right text-gray-600">{month.sessions}</td>
                    <td className="py-2 text-right text-gray-600">{month.avg_attendance}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Top Performing Classes */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Class Performance</h3>
            <Button variant="secondary" size="small" onClick={() => exportReport('performance')}>
              <Download className="h-3 w-3 mr-1" />
              Export
            </Button>
          </div>
          <div className="space-y-3">
            {reportData.class_stats
              .sort((a, b) => (b.present / b.total) - (a.present / a.total))
              .map((classData, index) => {
                const percentage = ((classData.present / classData.total) * 100).toFixed(1);
                return (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">{classData.class}</p>
                      <p className="text-sm text-gray-600">
                        {classData.present}/{classData.total} students
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-bold text-gray-900">{percentage}%</p>
                      <div className="w-16 bg-gray-200 rounded-full h-2 mt-1">
                        <div 
                          className="bg-blue-600 h-2 rounded-full" 
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </Card>
      </div>

      {/* Export Options */}
      <Card className="bg-gray-50 border-gray-200">
        <h4 className="font-medium text-gray-900 mb-3">Export Options</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Button variant="secondary" className="justify-start" onClick={() => exportReport('detailed')}>
            <Download className="h-4 w-4 mr-2" />
            Detailed Report (PDF)
          </Button>
          <Button variant="secondary" className="justify-start" onClick={() => exportReport('csv')}>
            <Download className="h-4 w-4 mr-2" />
            Raw Data (CSV)
          </Button>
          <Button variant="secondary" className="justify-start" onClick={() => exportReport('summary')}>
            <Download className="h-4 w-4 mr-2" />
            Summary Report (Excel)
          </Button>
        </div>
      </Card>
    </div>
  );
}
