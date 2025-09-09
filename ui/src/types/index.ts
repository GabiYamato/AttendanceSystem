export interface Student {
  student_id: string;
  name: string;
  face_registered: boolean;
  registered_at?: string;
}

export interface StudentRegistration {
  class_id: string;
  student_id: string;
  student_name: string;
}

export interface FaceRecognitionResponse {
  recognized: boolean;
  student_name?: string;
  confidence?: number;
  message: string;
  already_marked?: boolean;
}

export interface ApiResponse<T = any> {
  message: string;
  data?: T;
}
