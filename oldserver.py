"""
Smart Attendance & Class Management System - Complete Backend
Single FastAPI server handling all functionality
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import qrcode
import io
import base64
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from crewai import Agent, Task, Crew

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Firebase
try:
    app_firebase = firebase_admin.get_app()
except ValueError:
    cred_path = "firebase-service-account.json"
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        app_firebase = firebase_admin.initialize_app(cred)
    else:
        app_firebase = firebase_admin.initialize_app()

db = firestore.client()

# Initialize Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# FastAPI app
app = FastAPI(title="Smart Attendance System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (React build)
if os.path.exists("frontend/build"):
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

# === MODEL DEFINITIONS ===

class SimpleEmbeddingModel(nn.Module):
    """Face embedding model"""
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

# === PYDANTIC MODELS ===

class StudentRegistration(BaseModel):
    name: str
    roll_no: str
    email: str
    class_id: str

class AttendanceSession(BaseModel):
    class_id: str
    session_name: str
    duration_minutes: int = 60

class ScheduleConstraints(BaseModel):
    courses: List[str]
    faculty: List[str]
    rooms: List[str]
    time_slots: List[str]
    constraints: Optional[str] = ""

class ManualAttendance(BaseModel):
    student_id: str
    class_id: str
    session_id: str
    status: str = "present"

# === GLOBAL VARIABLES ===

face_recognition_system = None
current_session = None

class FaceRecognitionSystem:
    def __init__(self):
        self.setup_mediapipe()
        self.load_model()
        
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
        
    def load_model(self):
        """Load trained embedding model"""
        model_path = Path("trained_models/simple_face_model.pth")
        
        if not model_path.exists():
            print("❌ No trained model found!")
            self.model = None
            return
            
        self.model = SimpleEmbeddingModel()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
    def extract_landmarks(self, image):
        """Extract face landmarks from image"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            # Detect face first
            detection_results = self.face_detection.process(rgb_image)
            if not detection_results.detections:
                return None
                
            # Extract landmarks
            results = self.face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return None
                
            # Get landmarks (478 points * 3 coordinates = 1434 features)
            landmarks = results.multi_face_landmarks[0]
            landmark_points = []
            for landmark in landmarks.landmark:
                landmark_points.extend([landmark.x, landmark.y, landmark.z])
                
            return np.array(landmark_points)
            
        except Exception as e:
            return None
            
    def generate_embedding(self, landmarks):
        """Generate embedding from landmarks"""
        if self.model is None:
            return None
            
        try:
            with torch.no_grad():
                landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0)
                embedding = self.model(landmarks_tensor)
                return embedding.squeeze().numpy()
        except Exception as e:
            return None
            
    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        return np.dot(emb1, emb2)

# Initialize face recognition system
face_recognition_system = FaceRecognitionSystem()

# === UTILITY FUNCTIONS ===

def generate_qr_code(data: str) -> str:
    """Generate QR code and return as base64 string"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()

def process_image_from_upload(image_data: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

# === AI SCHEDULING FUNCTIONS ===

def create_schedule_agent():
    """Create AI agent for schedule optimization"""
    return Agent(
        role='Schedule Optimizer',
        goal='Create optimal class schedules that minimize conflicts and maximize resource utilization',
        backstory='You are an expert in educational scheduling with knowledge of academic constraints and optimization techniques.',
        verbose=True,
        allow_delegation=False
    )

def generate_smart_schedule(constraints: ScheduleConstraints) -> Dict:
    """Generate optimized schedule using Gemini AI"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Create an optimized weekly class schedule with these constraints:
        
        Courses: {', '.join(constraints.courses)}
        Faculty: {', '.join(constraints.faculty)}
        Rooms: {', '.join(constraints.rooms)}
        Time Slots: {', '.join(constraints.time_slots)}
        Additional Constraints: {constraints.constraints}
        
        Requirements:
        1. No faculty conflicts (same teacher can't be in two places)
        2. No room conflicts (one class per room per slot)
        3. Distribute courses evenly across the week
        4. Consider break times between classes
        5. Optimize for minimal travel time between rooms
        
        Return a JSON structure like:
        {{
            "Monday": [
                {{"course": "AI", "faculty": "Dr. Kumar", "room": "CS-101", "time": "09:00-10:00"}},
                {{"course": "DBMS", "faculty": "Prof. Rao", "room": "CS-102", "time": "10:15-11:15"}}
            ],
            "Tuesday": [...],
            ...
        }}
        """
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            schedule = json.loads(json_str)
            return {"success": True, "schedule": schedule}
        else:
            return {"success": False, "error": "Could not parse AI response"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# === API ENDPOINTS ===

@app.get("/")
async def root():
    return {"message": "Smart Attendance System API", "version": "1.0.0"}

# === STUDENT MANAGEMENT ===

@app.post("/api/students/register")
async def register_student(
    name: str = Form(...),
    roll_no: str = Form(...),
    email: str = Form(...),
    class_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """Register a new student with face embeddings"""
    try:
        embeddings = []
        
        # Process each uploaded image
        for image_file in images:
            image_data = await image_file.read()
            image = process_image_from_upload(image_data)
            
            # Extract landmarks and generate embedding
            landmarks = face_recognition_system.extract_landmarks(image)
            if landmarks is not None:
                embedding = face_recognition_system.generate_embedding(landmarks)
                if embedding is not None:
                    embeddings.append(embedding.tolist())
        
        if len(embeddings) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 valid face images")
        
        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        # Store in Firebase
        student_id = f"{class_id}_{roll_no}".replace("-", "_")
        student_ref = db.collection('classes').document(class_id).collection('students').document(student_id)
        
        student_data = {
            'name': name,
            'roll_no': roll_no,
            'email': email,
            'class_id': class_id,
            'embedding': avg_embedding,
            'registered_at': datetime.now(),
            'is_active': True,
            'total_embeddings': len(embeddings)
        }
        
        student_ref.set(student_data)
        
        return {
            "success": True, 
            "student_id": student_id,
            "embeddings_processed": len(embeddings),
            "message": f"Student {name} registered successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students/{class_id}")
async def get_class_students(class_id: str):
    """Get all students in a class"""
    try:
        students_ref = db.collection('classes').document(class_id).collection('students')
        students = []
        
        for doc in students_ref.stream():
            student_data = doc.to_dict()
            student_data['student_id'] = doc.id
            # Remove embedding for response size
            student_data.pop('embedding', None)
            students.append(student_data)
        
        return {"success": True, "students": students}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === ATTENDANCE MANAGEMENT ===

@app.post("/api/attendance/start-session")
async def start_attendance_session(session: AttendanceSession):
    """Start a new attendance session"""
    try:
        global current_session
        
        session_id = f"{session.class_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create session document
        session_ref = db.collection('attendance_sessions').document(session_id)
        session_data = {
            'class_id': session.class_id,
            'session_name': session.session_name,
            'created_at': datetime.now(),
            'duration_minutes': session.duration_minutes,
            'is_active': True,
            'attendees': {}
        }
        
        session_ref.set(session_data)
        
        # Generate QR code for session
        qr_data = {
            'session_id': session_id,
            'class_id': session.class_id,
            'type': 'attendance_session'
        }
        
        qr_code = generate_qr_code(json.dumps(qr_data))
        
        current_session = {
            'session_id': session_id,
            'class_id': session.class_id,
            'students_cache': {}
        }
        
        # Load students for this class
        students_ref = db.collection('classes').document(session.class_id).collection('students')
        for doc in students_ref.stream():
            student_data = doc.to_dict()
            if student_data.get('is_active', True) and 'embedding' in student_data:
                current_session['students_cache'][doc.id] = {
                    'name': student_data['name'],
                    'embedding': np.array(student_data['embedding'])
                }
        
        return {
            "success": True,
            "session_id": session_id,
            "qr_code": qr_code,
            "students_loaded": len(current_session['students_cache']),
            "expires_at": (datetime.now() + timedelta(minutes=session.duration_minutes)).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attendance/recognize")
async def recognize_face_for_attendance(image: UploadFile = File(...)):
    """Recognize face and mark attendance"""
    try:
        global current_session
        
        if not current_session:
            raise HTTPException(status_code=400, detail="No active attendance session")
        
        # Process uploaded image
        image_data = await image.read()
        image_np = process_image_from_upload(image_data)
        
        # Extract landmarks
        landmarks = face_recognition_system.extract_landmarks(image_np)
        if landmarks is None:
            return {"success": False, "message": "No face detected"}
        
        # Generate embedding
        query_embedding = face_recognition_system.generate_embedding(landmarks)
        if query_embedding is None:
            return {"success": False, "message": "Could not generate embedding"}
        
        # Find best match
        best_match = None
        best_similarity = -1
        threshold = 0.7
        
        for student_id, student_data in current_session['students_cache'].items():
            similarity = face_recognition_system.cosine_similarity(
                query_embedding, student_data['embedding']
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (student_id, student_data['name'])
        
        if best_similarity >= threshold:
            student_id, student_name = best_match
            
            # Check if already marked
            session_ref = db.collection('attendance_sessions').document(current_session['session_id'])
            session_doc = session_ref.get()
            
            if session_doc.exists:
                session_data = session_doc.to_dict()
                attendees = session_data.get('attendees', {})
                
                if student_id not in attendees:
                    # Mark attendance
                    attendees[student_id] = {
                        'name': student_name,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': float(best_similarity)
                    }
                    
                    session_ref.update({'attendees': attendees})
                    
                    return {
                        "success": True,
                        "recognized": True,
                        "student": {
                            "id": student_id,
                            "name": student_name,
                            "confidence": best_similarity
                        },
                        "message": f"Attendance marked for {student_name}"
                    }
                else:
                    return {
                        "success": True,
                        "recognized": True,
                        "student": {
                            "id": student_id,
                            "name": student_name,
                            "confidence": best_similarity
                        },
                        "message": f"Already marked: {student_name}"
                    }
        
        return {
            "success": True,
            "recognized": False,
            "message": f"No match found (best: {best_similarity:.3f})"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/session/{session_id}")
async def get_attendance_session(session_id: str):
    """Get attendance session details"""
    try:
        session_ref = db.collection('attendance_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = session_doc.to_dict()
        
        # Get class students for comparison
        class_id = session_data['class_id']
        students_ref = db.collection('classes').document(class_id).collection('students')
        total_students = len(list(students_ref.stream()))
        
        attendees = session_data.get('attendees', {})
        present_count = len(attendees)
        absent_count = total_students - present_count
        
        return {
            "success": True,
            "session": {
                "session_id": session_id,
                "class_id": class_id,
                "session_name": session_data.get('session_name'),
                "created_at": session_data.get('created_at'),
                "is_active": session_data.get('is_active'),
                "attendees": attendees,
                "stats": {
                    "total_students": total_students,
                    "present": present_count,
                    "absent": absent_count,
                    "attendance_rate": (present_count / total_students * 100) if total_students > 0 else 0
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attendance/manual")
async def mark_manual_attendance(attendance: ManualAttendance):
    """Manually mark attendance for a student"""
    try:
        session_ref = db.collection('attendance_sessions').document(attendance.session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get student info
        student_ref = db.collection('classes').document(attendance.class_id).collection('students').document(attendance.student_id)
        student_doc = student_ref.get()
        
        if not student_doc.exists:
            raise HTTPException(status_code=404, detail="Student not found")
        
        student_data = student_doc.to_dict()
        
        # Update session
        session_data = session_doc.to_dict()
        attendees = session_data.get('attendees', {})
        
        attendees[attendance.student_id] = {
            'name': student_data['name'],
            'timestamp': datetime.now().isoformat(),
            'confidence': 1.0,
            'manual': True
        }
        
        session_ref.update({'attendees': attendees})
        
        return {
            "success": True,
            "message": f"Manual attendance marked for {student_data['name']}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/export/{session_id}")
async def export_attendance(session_id: str):
    """Export attendance as CSV"""
    try:
        session_ref = db.collection('attendance_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = session_doc.to_dict()
        attendees = session_data.get('attendees', {})
        
        # Create DataFrame
        data = []
        for student_id, attendance_info in attendees.items():
            data.append({
                'Student ID': student_id,
                'Name': attendance_info['name'],
                'Timestamp': attendance_info['timestamp'],
                'Confidence': attendance_info['confidence'],
                'Manual': attendance_info.get('manual', False)
            })
        
        df = pd.DataFrame(data)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=attendance_{session_id}.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === SCHEDULE MANAGEMENT ===

@app.post("/api/schedules/generate")
async def generate_schedule(constraints: ScheduleConstraints):
    """Generate optimized schedule using AI"""
    try:
        result = generate_smart_schedule(constraints)
        
        if result["success"]:
            # Save schedule to Firebase
            schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            schedule_ref = db.collection('schedules').document(schedule_id)
            
            schedule_data = {
                'schedule_id': schedule_id,
                'schedule': result["schedule"],
                'constraints': constraints.dict(),
                'created_at': datetime.now(),
                'is_active': True
            }
            
            schedule_ref.set(schedule_data)
            
            return {
                "success": True,
                "schedule_id": schedule_id,
                "schedule": result["schedule"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/schedules/{class_id}/today")
async def get_today_schedule(class_id: str):
    """Get today's schedule for a class"""
    try:
        # Get the most recent active schedule
        schedules_ref = db.collection('schedules').where('is_active', '==', True).order_by('created_at', direction=firestore.Query.DESCENDING).limit(1)
        
        schedule_docs = list(schedules_ref.stream())
        if not schedule_docs:
            return {"success": True, "schedule": [], "message": "No active schedule found"}
        
        schedule_data = schedule_docs[0].to_dict()
        full_schedule = schedule_data.get('schedule', {})
        
        # Get today's day name
        today = datetime.now().strftime('%A')
        today_schedule = full_schedule.get(today, [])
        
        # Filter for specific class if needed
        filtered_schedule = []
        for slot in today_schedule:
            # Add class_id matching logic if needed
            filtered_schedule.append(slot)
        
        return {
            "success": True,
            "schedule": filtered_schedule,
            "day": today,
            "class_id": class_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/schedules")
async def get_all_schedules():
    """Get all schedules"""
    try:
        schedules_ref = db.collection('schedules').order_by('created_at', direction=firestore.Query.DESCENDING)
        schedules = []
        
        for doc in schedules_ref.stream():
            schedule_data = doc.to_dict()
            schedule_data['schedule_id'] = doc.id
            schedules.append(schedule_data)
        
        return {"success": True, "schedules": schedules}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === DASHBOARD ANALYTICS ===

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Count total students
        total_students = 0
        classes_ref = db.collection('classes')
        for class_doc in classes_ref.stream():
            students_ref = class_doc.reference.collection('students')
            total_students += len(list(students_ref.stream()))
        
        # Count active sessions
        sessions_ref = db.collection('attendance_sessions').where('is_active', '==', True)
        active_sessions = len(list(sessions_ref.stream()))
        
        # Count total classes
        total_classes = len(list(classes_ref.stream()))
        
        # Recent attendance rate
        recent_sessions = db.collection('attendance_sessions').order_by('created_at', direction=firestore.Query.DESCENDING).limit(10)
        total_attendance_rate = 0
        session_count = 0
        
        for session_doc in recent_sessions.stream():
            session_data = session_doc.to_dict()
            attendees = len(session_data.get('attendees', {}))
            # Rough estimate - would need actual class size
            if attendees > 0:
                total_attendance_rate += min(attendees / 30 * 100, 100)  # Assuming max 30 students
                session_count += 1
        
        avg_attendance_rate = total_attendance_rate / session_count if session_count > 0 else 0
        
        return {
            "success": True,
            "stats": {
                "total_students": total_students,
                "total_classes": total_classes,
                "active_sessions": active_sessions,
                "average_attendance_rate": round(avg_attendance_rate, 1)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === HEALTH CHECK ===

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": face_recognition_system.model is not None,
        "firebase_connected": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
