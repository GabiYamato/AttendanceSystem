"""
Hackathon Face Recognition System
Comprehensive student registration and live detection using face_recognition library
Integrated with Firebase for storage
Features: Multi-face detection, real-time preview, robust registration with multiple captures
No attendance marking - pure recognition demo
"""

import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os
import argparse
import time
import firebase_admin
from firebase_admin import credentials, firestore

class HackathonFirebase:
    """Firebase manager adapted for hackathon face recognition"""
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

    def register_student(self, class_id, student_id, name, encodings):
        """Register student with face encodings"""
        try:
            # Store student info
            student_ref = self.db.collection('classes').document(class_id).collection('students').document(student_id)
            student_ref.set({
                'name': name,
                'class_id': class_id,
                'registered_at': datetime.now(),
                'is_active': True
            })

            # Store encodings
            for i, encoding in enumerate(encodings):
                encoding_ref = student_ref.collection('encodings').document(f'frame_{i}')
                encoding_ref.set({
                    'encoding': encoding.tolist(),
                    'frame_number': i,
                    'created_at': datetime.now()
                })

            print(f"✅ Registered {name} with {len(encodings)} encodings")
            return True

        except Exception as e:
            print(f"❌ Registration failed: {e}")
            return False

    def get_class_students(self, class_id):
        """Get all students and their encodings for a class"""
        try:
            students = {}
            students_ref = self.db.collection('classes').document(class_id).collection('students')

            for student_doc in students_ref.stream():
                student_data = student_doc.to_dict()
                if not student_data.get('is_active', True):
                    continue

                student_id = student_doc.id

                # Get encodings
                encodings = []
                encodings_ref = student_doc.reference.collection('encodings')
                for enc_doc in encodings_ref.stream():
                    enc_data = enc_doc.to_dict()
                    encodings.append(np.array(enc_data['encoding']))

                if encodings:
                    students[student_id] = {
                        'name': student_data['name'],
                        'encodings': encodings
                    }

            return students

        except Exception as e:
            print(f"❌ Error loading students: {e}")
            return {}

class HackathonFaceRecognizer:
    def __init__(self):
        """Initialize the face recognition system"""
        self.firebase = HackathonFirebase()
        self.students_cache = {}
        self.frame_count = 0

    def register_student(self, class_id, student_id, student_name, num_captures=10):
        """Register a new student with multiple face captures for robustness"""
        print(f"\n🎯 Registering: {student_name} ({student_id}) in {class_id}")
        print("📷 Starting camera for registration...")
        print(f"📋 Capturing {num_captures} frames manually - look at camera, vary poses slightly")
        print("   • Press SPACEBAR to capture a frame")
        print("   • Press 'q' to finish early or cancel")
        print("   • Camera running at 24 FPS")

        cap = cv2.VideoCapture(0)
        # Set camera to 24 FPS
        cap.set(cv2.CAP_PROP_FPS, 24)
        encodings = []
        frame_delay = int(1000 / 24)  # ~42ms delay for 24 FPS

        while len(encodings) < num_captures:
            ret, frame = cap.read()
            if not ret:
                break

            # Display preview with instructions
            preview = frame.copy()
            cv2.putText(preview, f"Captures: {len(encodings)}/{num_captures}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(preview, "Press SPACEBAR to capture", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(preview, "Press 'q' to quit", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Registration', preview)

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord(' '):  # Spacebar pressed
                # Find face locations and encodings
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                if face_encodings:
                    encodings.append(face_encodings[0])  # Take the first face
                    print(f"✅ Capture {len(encodings)}/{num_captures}")
                else:
                    print("⚠️ No face detected - try again")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)

        if len(encodings) >= 5:  # Minimum for reliability
            success = self.firebase.register_student(class_id, student_id, student_name, encodings)
            if success:
                print(f"✅ Registration complete with {len(encodings)} encodings")
            return success
        else:
            print("❌ Registration failed - insufficient quality captures")
            return False

    def load_class_students(self, class_id):
        """Load students for recognition"""
        self.students_cache = self.firebase.get_class_students(class_id)
        print(f"✅ Loaded {len(self.students_cache)} students for recognition")

    def live_detection(self, class_id, tolerance=0.5):
        """Live face detection and recognition preview"""
        print(f"\n🎯 Starting live detection for {class_id}")
        print(f"🔍 Tolerance: {tolerance} (lower = stricter)")
 
        # Load students
        self.load_class_students(class_id)
        if not self.students_cache:
            print("❌ No students registered")
            return

        # Prepare known encodings and names
        known_encodings = []
        known_names = []
        for student_id, data in self.students_cache.items():
            for encoding in data['encodings']:
                known_encodings.append(encoding)
                known_names.append(data['name'])

        cap = cv2.VideoCapture(0)
        # Set camera to 24 FPS
        cap.set(cv2.CAP_PROP_FPS, 24)
        # frame_delay = int(1000 / 24)  # ~42ms delay for 24 FPS

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Find all faces in the frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Recognize each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Live Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.load_class_students(class_id)

        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Hackathon Face Recognition System")
    parser.add_argument('--mode', choices=['register', 'detect'], required=True,
                        help="Mode: 'register' or 'detect'")
    parser.add_argument('--class_id', default='hackathon_class', help="Class ID")
    parser.add_argument('--student_id', help="Student ID (for registration)")
    parser.add_argument('--name', help="Student name (for registration)")
    parser.add_argument('--tolerance', type=float, default=0.5, help="Recognition tolerance")

    args = parser.parse_args()
    system = HackathonFaceRecognizer()

    if args.mode == 'register':
        if not args.student_id or not args.name:
            print("❌ Provide --student_id and --name for registration")
            return
        system.register_student(args.class_id, args.student_id, args.name)
    elif args.mode == 'detect':
        system.live_detection(args.class_id, args.tolerance)

if __name__ == "__main__":
    main()
