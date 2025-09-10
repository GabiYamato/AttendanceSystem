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
                if last_time and hasattr(last_time, 'replace'):  # Check if it's a datetime object
                    # Convert to timezone-naive datetime for comparison
                    if last_time.tzinfo is not None:
                        last_time = last_time.replace(tzinfo=None)
                    if last_time > cutoff_time:
                        return True
            return False
            
        except Exception as e:
            print(f"❌ Error checking attendance: {e}")
            return False  # Default to allowing attendance marking if there's an error

    def list_class_students(self, class_id):
        """List all students in a class"""
        try:
            students = []
            students_ref = self.db.collection('classes').document(class_id).collection('students')
            
            for student_doc in students_ref.stream():
                student_data = student_doc.to_dict()
                students.append({
                    'student_id': student_doc.id,
                    'name': student_data.get('name', 'Unknown'),
                    'registered_at': student_data.get('registered_at'),
                    'is_active': student_data.get('is_active', True)
                })
            
            return students
            
        except Exception as e:
            print(f"❌ Error listing students: {e}")
            return []

    def delete_student(self, class_id, student_id):
        """Delete a specific student and all their data"""
        try:
            student_ref = self.db.collection('classes').document(class_id).collection('students').document(student_id)
            
            # Delete embeddings subcollection
            embeddings_ref = student_ref.collection('embeddings')
            for emb_doc in embeddings_ref.stream():
                emb_doc.reference.delete()
            
            # Delete attendance subcollection
            attendance_ref = student_ref.collection('attendance')
            for att_doc in attendance_ref.stream():
                att_doc.reference.delete()
            
            # Delete student document
            student_ref.delete()
            
            print(f"✅ Deleted student {student_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting student {student_id}: {e}")
            return False

    def clear_class_data(self, class_id):
        """Clear all student data for a class"""
        try:
            students_ref = self.db.collection('classes').document(class_id).collection('students')
            
            deleted_count = 0
            for student_doc in students_ref.stream():
                student_id = student_doc.id
                if self.delete_student(class_id, student_id):
                    deleted_count += 1
            
            print(f"✅ Cleared {deleted_count} students from class {class_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing class data: {e}")
            return False
