import React, { useState, useEffect } from 'react';
import { Card, Button } from '../components/UI';
import { 
  Users, 
  Calendar, 
  TrendingUp, 
  Activity,
  Clock,
  BookOpen,
  UserCheck
} from 'lucide-react';

export function Dashboard() {
  const [stats, setStats] = useState({
    total_students: 0,
    total_classes: 0,
    active_sessions: 0,
    average_attendance_rate: 0
  });
  const [loading, setLoading] = useState(true);
  const [todaySchedule, setTodaySchedule] = useState([]);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch dashboard stats
      const statsResponse = await fetch('/api/dashboard/stats');
      const statsData = await statsResponse.json();
      
      if (statsData.success) {
        setStats(statsData.stats);
      }

      // Fetch today's schedule
      const scheduleResponse = await fetch('/api/schedules/default/today');
      const scheduleData = await scheduleResponse.json();
      
      if (scheduleData.success) {
        setTodaySchedule(scheduleData.schedule);
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Students',
      value: stats.total_students,
      icon: Users,
      color: 'blue',
      change: '+12%'
    },
    {
      title: 'Active Classes',
      value: stats.total_classes,
      icon: BookOpen,
      color: 'green',
      change: '+5%'
    },
    {
      title: 'Live Sessions',
      value: stats.active_sessions,
      icon: Activity,
      color: 'yellow',
      change: 'Now'
    },
    {
      title: 'Avg Attendance',
      value: `${stats.average_attendance_rate}%`,
      icon: TrendingUp,
      color: 'purple',
      change: '+2.1%'
    }
  ];

  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    yellow: 'bg-yellow-100 text-yellow-600',
    purple: 'bg-purple-100 text-purple-600'
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="animate-pulse">
              <div className="h-20 bg-gray-200 rounded"></div>
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
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600">Welcome to Smart Attendance System</p>
        </div>
        <div className="flex space-x-3">
          <Button variant="secondary">
            <Calendar className="h-4 w-4 mr-2" />
            View Schedule
          </Button>
          <Button>
            <UserCheck className="h-4 w-4 mr-2" />
            Start Session
          </Button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} className="relative overflow-hidden">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                  <p className="text-xs text-green-600 mt-1">{stat.change}</p>
                </div>
                <div className={`p-3 rounded-lg ${colorClasses[stat.color]}`}>
                  <Icon className="h-6 w-6" />
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Today's Schedule */}
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Today's Schedule</h3>
            <Clock className="h-5 w-5 text-gray-400" />
          </div>
          
          {todaySchedule.length > 0 ? (
            <div className="space-y-3">
              {todaySchedule.map((slot, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">{slot.course}</p>
                    <p className="text-sm text-gray-600">{slot.faculty} • {slot.room}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">{slot.time}</p>
                    <Button size="small" className="mt-1">Take Attendance</Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <Calendar className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No classes scheduled for today</p>
            </div>
          )}
        </Card>

        {/* Quick Actions */}
        <Card>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <Button className="w-full justify-start">
              <UserCheck className="h-4 w-4 mr-2" />
              Start Attendance Session
            </Button>
            <Button variant="secondary" className="w-full justify-start">
              <Users className="h-4 w-4 mr-2" />
              Register New Student
            </Button>
            <Button variant="secondary" className="w-full justify-start">
              <Calendar className="h-4 w-4 mr-2" />
              Generate Schedule
            </Button>
            <Button variant="secondary" className="w-full justify-start">
              <TrendingUp className="h-4 w-4 mr-2" />
              View Reports
            </Button>
          </div>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
        <div className="space-y-4">
          {[
            { action: 'New student registered', student: 'John Smith', time: '2 minutes ago' },
            { action: 'Attendance session completed', class: 'CS-301', time: '15 minutes ago' },
            { action: 'Schedule generated', type: 'Weekly Schedule', time: '1 hour ago' },
          ].map((activity, index) => (
            <div key={index} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
              <div>
                <p className="font-medium text-gray-900">{activity.action}</p>
                <p className="text-sm text-gray-600">
                  {activity.student || activity.class || activity.type}
                </p>
              </div>
              <p className="text-sm text-gray-500">{activity.time}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
