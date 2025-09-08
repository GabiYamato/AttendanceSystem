"""
FastAPI Face Recognition Attendance System
Handles student registration, live attendance, and data retrieval
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import base64
import io
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import os
from typing import List, Optional
import json

app = FastAPI(title="Face Recognition Attendance API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class StudentRegistration(BaseModel):
    class_id: str
    student_id: str
    student_name: str

class AttendanceRequest(BaseModel):
    class_id: str
    image_data: str  # Base64 encoded image

class AttendanceRecord(BaseModel):
    student_id: str
    student_name: str
    timestamp: datetime
    confidence: float
    status: str

# Firebase and ML setup
class SimpleEmbeddingModel(nn.Module):
    """Same model as training script"""
    def __init__(self, input_dim=1434, embedding_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        return F.normalize(self.layers(x), p=2, dim=1)

class FaceRecognitionService:
    def __init__(self):
        self.setup_firebase()
        self.setup_mediapipe()
        self.load_model()
        
    def setup_firebase(self):
        """Initialize Firebase connection"""
        try:
            self.app = firebase_admin.get_app()
        except ValueError:
            cred_path = "firebase-service-account.json"
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                self.app = firebase_admin.initialize_app(cred)
            else:
                self.app = firebase_admin.initialize_app()
        self.db = firestore.client()
        print("✅ Firebase connected")

    def setup_mediapipe(self):
        """Setup MediaPipe components"""
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe initialized")

    def load_model(self):
        """Load trained embedding model"""
        model_path = "trained_models/simple_face_model.pth"
        if os.path.exists(model_path):
            self.model = SimpleEmbeddingModel()
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print("✅ Model loaded")
        else:
            print("⚠️ No trained model found, using basic face detection")
            self.model = None

    def extract_landmarks(self, image):
        """Extract face landmarks from image"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detection_results = self.face_detection.process(rgb_image)
            
            if not detection_results.detections:
                return None
                
            results = self.face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0]
            landmark_points = []
            for landmark in landmarks.landmark:
                landmark_points.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmark_points)
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None

    def generate_embedding(self, landmarks):
        """Generate embedding from landmarks"""
        if self.model is None:
            return landmarks[:128]  # Fallback to first 128 landmarks
        
        try:
            with torch.no_grad():
                landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0)
                embedding = self.model(landmarks_tensor)
                return embedding.squeeze().numpy()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def base64_to_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None

# Initialize service
face_service = FaceRecognitionService()

@app.on_event("startup")
async def startup_event():
    print("🚀 Face Recognition Attendance API Started")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Face Recognition Attendance System API", "status": "active"}

@app.post("/api/register-student")
async def register_student(registration: StudentRegistration):
    """Register a new student (without face data initially)"""
    try:
        student_ref = face_service.db.collection('classes').document(registration.class_id).collection('students').document(registration.student_id)
        student_ref.set({
            'name': registration.student_name,
            'class_id': registration.class_id,
            'registered_at': datetime.now(),
            'is_active': True,
            'face_registered': False
        })
        
        return {"message": f"Student {registration.student_name} registered successfully", "student_id": registration.student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/register-face")
async def register_face(class_id: str, student_id: str, image_data: str):
    """Register face data for a student"""
    try:
        # Convert base64 to image
        image = face_service.base64_to_image(image_data)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Extract landmarks and generate embedding
        landmarks = face_service.extract_landmarks(image)
        if landmarks is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        embedding = face_service.generate_embedding(landmarks)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Failed to generate face embedding")
        
        # Store embedding in Firebase
        student_ref = face_service.db.collection('classes').document(class_id).collection('students').document(student_id)
        embedding_ref = student_ref.collection('face_embeddings').document(f'embedding_{datetime.now().timestamp()}')
        
        embedding_ref.set({
            'embedding': embedding.tolist(),
            'registered_at': datetime.now()
        })
        
        # Update student to mark face as registered
        student_ref.update({'face_registered': True})
        
        return {"message": "Face registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face registration failed: {str(e)}")

@app.post("/api/mark-attendance")
async def mark_attendance(request: AttendanceRequest):
    """Mark attendance using face recognition"""
    try:
        # Convert base64 to image
        image = face_service.base64_to_image(request.image_data)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Extract landmarks from current image
        current_landmarks = face_service.extract_landmarks(image)
        if current_landmarks is None:
            return {"recognized": False, "message": "No face detected"}
        
        current_embedding = face_service.generate_embedding(current_landmarks)
        if current_embedding is None:
            return {"recognized": False, "message": "Failed to process face"}
        
        # Get all students in class
        students_ref = face_service.db.collection('classes').document(request.class_id).collection('students')
        students = students_ref.where('face_registered', '==', True).stream()
        
        best_match = None
        best_similarity = -1
        threshold = 0.7
        
        for student_doc in students:
            student_data = student_doc.to_dict()
            student_id = student_doc.id
            
            # Get student's face embeddings
            embeddings_ref = student_doc.reference.collection('face_embeddings')
            embeddings = embeddings_ref.stream()
            
            for emb_doc in embeddings:
                emb_data = emb_doc.to_dict()
                stored_embedding = np.array(emb_data['embedding'])
                
                similarity = face_service.cosine_similarity(current_embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'student_id': student_id,
                        'student_name': student_data['name'],
                        'confidence': similarity
                    }
        
        if best_match and best_similarity >= threshold:
            # Check if already marked today
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_ref = (face_service.db.collection('classes')
                            .document(request.class_id)
                            .collection('attendance')
                            .document(today)
                            .collection('records')
                            .document(best_match['student_id']))
            
            existing = attendance_ref.get()
            if existing.exists:
                return {
                    "recognized": True,
                    "already_marked": True,
                    "student_name": best_match['student_name'],
                    "message": "Attendance already marked today"
                }
            
            # Mark attendance
            attendance_ref.set({
                'student_id': best_match['student_id'],
                'student_name': best_match['student_name'],
                'timestamp': datetime.now(),
                'confidence': float(best_similarity),
                'status': 'present'
            })
            
            return {
                "recognized": True,
                "student_name": best_match['student_name'],
                "confidence": float(best_similarity),
                "message": "Attendance marked successfully"
            }
        else:
            return {
                "recognized": False,
                "confidence": float(best_similarity) if best_match else 0.0,
                "message": "Face not recognized"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attendance marking failed: {str(e)}")

@app.get("/api/classes/{class_id}/students")
async def get_class_students(class_id: str):
    """Get all students in a class"""
    try:
        students_ref = face_service.db.collection('classes').document(class_id).collection('students')
        students = []
        
        for student_doc in students_ref.stream():
            student_data = student_doc.to_dict()
            students.append({
                'student_id': student_doc.id,
                'name': student_data['name'],
                'face_registered': student_data.get('face_registered', False),
                'registered_at': student_data['registered_at'].isoformat() if student_data.get('registered_at') else None
            })
        
        return {"students": students}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch students: {str(e)}")

@app.get("/api/classes/{class_id}/attendance")
async def get_class_attendance(class_id: str, date: Optional[str] = None):
    """Get attendance for a class on a specific date"""
    try:
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        attendance_ref = (face_service.db.collection('classes')
                         .document(class_id)
                         .collection('attendance')
                         .document(date)
                         .collection('records'))
        
        attendance_records = []
        for record_doc in attendance_ref.stream():
            record_data = record_doc.to_dict()
            attendance_records.append({
                'student_id': record_data['student_id'],
                'student_name': record_data['student_name'],
                'timestamp': record_data['timestamp'].isoformat(),
                'confidence': record_data['confidence'],
                'status': record_data['status']
            })
        
        return {
            "date": date,
            "class_id": class_id,
            "attendance": attendance_records,
            "total_present": len(attendance_records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch attendance: {str(e)}")

@app.get("/api/classes")
async def get_all_classes():
    """Get all classes"""
    try:
        classes_ref = face_service.db.collection('classes')
        classes = []
        
        for class_doc in classes_ref.stream():
            class_id = class_doc.id
            # Count students
            students_count = len(list(class_doc.reference.collection('students').stream()))
            
            classes.append({
                'class_id': class_id,
                'students_count': students_count
            })
        
        return {"classes": classes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch classes: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
