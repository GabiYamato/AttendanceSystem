"""
Simple Firebase Manager for Face Recognition Attendance System
Single file for all Firebase operations
"""
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from datetime import datetime
import os

class SimpleFirebase:
    def __init__(self):
        """Initialize Firebase connection"""
        try:
            # Try to get existing app
            self.app = firebase_admin.get_app()
        except ValueError:
            # Initialize new app
            cred_path = "firebase-service-account.json"
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                self.app = firebase_admin.initialize_app(cred)
            else:
                # Use default credentials
                self.app = firebase_admin.initialize_app()
                
        self.db = firestore.client()
        print("✅ Firebase connected")

    def register_student(self, class_id, student_id, name, embeddings):
        """Register student with embeddings"""
        try:
            # Store student info
            student_ref = self.db.collection('classes').document(class_id).collection('students').document(student_id)
            student_ref.set({
                'name': name,
                'class_id': class_id,
                'registered_at': datetime.now(),
                'is_active': True
            })
            
            # Store embeddings
            for i, embedding in enumerate(embeddings):
                embedding_ref = student_ref.collection('embeddings').document(f'frame_{i}')
                embedding_ref.set({
                    'embedding': embedding.tolist(),
                    'frame_number': i,
                    'created_at': datetime.now()
                })
            
            print(f"✅ Registered {name} with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            return False

    def get_class_students(self, class_id):
        """Get all students and their embeddings for a class"""
        try:
            students = {}
            students_ref = self.db.collection('classes').document(class_id).collection('students')
            
            for student_doc in students_ref.stream():
                student_data = student_doc.to_dict()
                if not student_data.get('is_active', True):
                    continue
                    
                student_id = student_doc.id
                
                # Get embeddings
                embeddings = []
                embeddings_ref = student_doc.reference.collection('embeddings')
                for emb_doc in embeddings_ref.stream():
                    emb_data = emb_doc.to_dict()
                    embeddings.append(np.array(emb_data['embedding']))
                
                if embeddings:
                    students[student_id] = {
                        'name': student_data['name'],
                        'embeddings': embeddings
                    }
            
            return students
            
        except Exception as e:
            print(f"❌ Error loading students: {e}")
            return {}

    def mark_attendance(self, class_id, student_id, confidence):
        """Mark attendance for student"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_ref = (self.db.collection('classes')
                            .document(class_id)
                            .collection('students')
                            .document(student_id)
                            .collection('attendance')
                            .document(today))
            
            attendance_ref.set({
                'date': today,
                'timestamp': datetime.now(),
                'confidence': confidence,
                'status': 'present'
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Attendance marking failed: {e}")
            return False

    def check_recent_attendance(self, class_id, student_id, minutes=5):
        """Check if student was marked present recently"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_ref = (self.db.collection('classes')
                            .document(class_id)
                            .collection('students')
                            .document(student_id)
                            .collection('attendance')
                            .document(today))
            
            doc = attendance_ref.get()
            if doc.exists:
                data = doc.to_dict()
                last_time = data.get('timestamp')
                if last_time and last_time > cutoff_time:
                    return True
            return False
            
        except Exception as e:
            print(f"❌ Error checking attendance: {e}")
            return False
