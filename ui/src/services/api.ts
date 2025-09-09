import type { FaceRecognitionResponse, StudentRegistration, Student, ApiResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Network error' }));
      throw new Error(error.detail || 'Request failed');
    }

    return response.json();
  }

  async registerStudent(data: StudentRegistration): Promise<ApiResponse> {
    return this.request('/api/register-student', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async registerFace(classId: string, studentId: string, imageData: string): Promise<ApiResponse> {
    const formData = new FormData();
    formData.append('class_id', classId);
    formData.append('student_id', studentId);
    formData.append('image_data', imageData);

    const response = await fetch(`${API_BASE_URL}/api/register-face`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Network error' }));
      throw new Error(error.detail || 'Face registration failed');
    }

    return response.json();
  }

  async markAttendance(classId: string, imageData: string): Promise<FaceRecognitionResponse> {
    return this.request<FaceRecognitionResponse>('/api/mark-attendance', {
      method: 'POST',
      body: JSON.stringify({
        class_id: classId,
        image_data: imageData,
      }),
    });
  }

  async getClassStudents(classId: string): Promise<{ students: Student[] }> {
    return this.request<{ students: Student[] }>(`/api/classes/${classId}/students`);
  }
}

export const apiService = new ApiService();
