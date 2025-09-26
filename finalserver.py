"""
Streamlit Attendance System
Step-by-step attendance marking with QR code verification and face recognition
1. Scan QR code to verify class
2. Face recognition for student identification
3. Mark attendance in Firebase
"""

import streamlit as st
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os
import qrcode
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, firestore
import time

class AttendanceFirebase:
    """Firebase manager for attendance system"""
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

    def verify_class_code(self, class_id):
        """Verify if class exists and is active, create if doesn't exist"""
        try:
            class_ref = self.db.collection('classes').document(class_id)
            class_doc = class_ref.get()
            
            if class_doc.exists:
                class_data = class_doc.to_dict()
                return class_data.get('is_active', True)
            else:
                # Create the class if it doesn't exist
                class_ref.set({
                    'name': f'Class {class_id}',
                    'created_at': datetime.now(),
                    'is_active': True,
                    'description': f'Auto-created class for {class_id}'
                })
                st.info(f"ℹ️ Created new class: {class_id}")
                return True
        except Exception as e:
            st.error(f"Error verifying class: {e}")
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
            st.error(f"Error loading students: {e}")
            return {}

    def mark_attendance(self, class_id, student_id, student_name):
        """Mark attendance for a student"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_ref = (self.db.collection('classes')
                             .document(class_id)
                             .collection('attendance')
                             .document(today)
                             .collection('students')
                             .document(student_id))
            
            attendance_data = {
                'student_name': student_name,
                'marked_at': datetime.now(),
                'status': 'present',
                'marked_by': 'face_recognition'
            }
            
            attendance_ref.set(attendance_data)
            return True
            
        except Exception as e:
            st.error(f"Error marking attendance: {e}")
            return False

    def check_attendance_today(self, class_id, student_id):
        """Check if student already marked attendance today"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_ref = (self.db.collection('classes')
                             .document(class_id)
                             .collection('attendance')
                             .document(today)
                             .collection('students')
                             .document(student_id))
            
            doc = attendance_ref.get()
            return doc.exists
            
        except Exception as e:
            st.error(f"Error checking attendance: {e}")
            return False

class QRCodeManager:
    """QR Code scanning utilities"""
    
    @staticmethod
    def scan_qr_from_camera():
        """Scan QR code from camera using OpenCV"""
        cap = cv2.VideoCapture(0)
        scanned_data = None
        
        # Create QR code detector
        qr_detector = cv2.QRCodeDetector()
        
        # Create placeholder for camera feed
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        status_placeholder.info("📷 Camera active - Point QR code at camera")
        
        # Add stop button
        stop_button = st.button("Stop QR Scanning")
        
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        
        while scanned_data is None and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check for timeout
            if time.time() - start_time > timeout:
                status_placeholder.warning("⏰ QR scan timeout - please try again")
                break
            
            # Detect and decode QR codes using OpenCV
            retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)
            
            if retval:
                for i, info in enumerate(decoded_info):
                    if info and info.startswith("CLASS:"):
                        scanned_data = info.replace("CLASS:", "")
                        status_placeholder.success(f"✅ QR Code detected: {scanned_data}")
                        
                        # Draw rectangle around QR code
                        if points is not None and len(points) > i:
                            pts = points[i].astype(int)
                            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                            cv2.putText(frame, f"Class: {scanned_data}", 
                                       tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 255, 0), 2)
                        break
            
            # Convert frame to RGB for streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", width=640)
            
            # Small delay to prevent overwhelming the browser
            time.sleep(0.1)
        
        cap.release()
        camera_placeholder.empty()
        
        return scanned_data

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'class_id' not in st.session_state:
        st.session_state.class_id = None
    if 'firebase' not in st.session_state:
        st.session_state.firebase = AttendanceFirebase()
    if 'students_cache' not in st.session_state:
        st.session_state.students_cache = {}

def step1_qr_scan():
    """Step 1: QR Code Scanning"""
    st.header("📱 Step 1: Scan Class QR Code")
    
    st.subheader("Scan QR Code")
    st.info("📷 Point your camera at the class QR code to begin attendance")
    
    if st.button("Start QR Camera Scan", type="primary", use_container_width=True):
        scanned_class_id = QRCodeManager.scan_qr_from_camera()
        
        if scanned_class_id:
            if st.session_state.firebase.verify_class_code(scanned_class_id):
                st.session_state.class_id = scanned_class_id
                st.session_state.step = 2
                st.success(f"✅ Class verified: {scanned_class_id}")
                st.balloons()
                time.sleep(1)  # Brief pause before auto-starting face recognition
                st.rerun()
            else:
                st.error("❌ Error verifying class code")
        else:
            st.warning("⚠️ No QR code detected - please try again")
    
    st.divider()
    st.caption("💡 Make sure the QR code is well-lit and clearly visible to the camera")

def step2_face_recognition():
    """Step 2: Face Recognition - Auto-starts"""
    st.header(f"👤 Step 2: Face Recognition - Class: {st.session_state.class_id}")
    
    # Load students if not already loaded
    if not st.session_state.students_cache:
        with st.spinner("Loading registered students..."):
            st.session_state.students_cache = st.session_state.firebase.get_class_students(st.session_state.class_id)
    
    if not st.session_state.students_cache:
        st.error("❌ No students registered for this class")
        st.info("📝 Register students first using: `python3 face_recognizer.py --mode register --class_id {} --student_id <ID> --name <NAME>`".format(st.session_state.class_id))
        if st.button("← Back to QR Scan"):
            st.session_state.step = 1
            st.rerun()
        return
    
    st.success(f"📊 {len(st.session_state.students_cache)} students registered - Starting face recognition...")
    
    # Auto-start face recognition
    if 'recognition_started' not in st.session_state:
        st.session_state.recognition_started = True
    
    # Prepare known encodings and names
    known_encodings = []
    known_names = []
    student_mapping = {}
    
    for student_id, data in st.session_state.students_cache.items():
        for encoding in data['encodings']:
            known_encodings.append(encoding)
            known_names.append(data['name'])
            student_mapping[data['name']] = student_id
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Camera feed placeholder
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("Controls")
        
        if st.button("← Back to QR Scan"):
            st.session_state.step = 1
            st.session_state.class_id = None
            st.session_state.students_cache = {}
            st.session_state.recognition_started = False
            st.rerun()
        
        st.subheader("Settings")
        tolerance = st.slider("Recognition Tolerance", 0.3, 0.8, 0.5, 0.05)
        st.caption("Lower = Stricter recognition")
        
        st.subheader("Status")
        attendance_status = st.empty()
    
    # Auto-start face recognition
    if st.session_state.recognition_started:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 24)
        
        status_placeholder.info("🎥 Face recognition active - Look at the camera")
        
        detection_count = {}  # Track consecutive detections
        required_detections = 5  # Require 5 consecutive detections
        marked_students = set()  # Track students already marked today
        
        frame_count = 0
        max_frames = 3000  # 2-3 minutes at 24 FPS
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                # Find all faces in the frame
                face_locations = face_recognition.face_locations(frame, model='hog')
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                detected_students = []
                
                # Recognize each face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                    name = "Unknown"
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            detected_students.append(name)
                    
                    # Draw box and label
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                
                # Track consecutive detections
                for student in detected_students:
                    if student not in marked_students:
                        detection_count[student] = detection_count.get(student, 0) + 1
                        
                        if detection_count[student] >= required_detections:
                            student_id = student_mapping[student]
                            
                            # Check if already marked today
                            if not st.session_state.firebase.check_attendance_today(st.session_state.class_id, student_id):
                                # Mark attendance
                                if st.session_state.firebase.mark_attendance(st.session_state.class_id, student_id, student):
                                    marked_students.add(student)
                                    attendance_status.success(f"✅ {student} - Present")
                                    st.balloons()
                                else:
                                    attendance_status.error(f"❌ Failed to mark {student}")
                            else:
                                marked_students.add(student)
                                attendance_status.info(f"ℹ️ {student} - Already marked present")
                            
                            # Reset counter
                            detection_count[student] = 0
                
                # Reset counters for students not detected in current frame
                for student in list(detection_count.keys()):
                    if student not in detected_students:
                        detection_count[student] = max(0, detection_count[student] - 1)
            
            # Convert frame to RGB for streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", width=640)
            
            # Small delay
            time.sleep(0.033)  # ~30 FPS display
        
        cap.release()
        camera_placeholder.empty()
        status_placeholder.success("🎥 Face recognition completed")
        
        # Show final summary
        if marked_students:
            st.success(f"🎯 Attendance marked for {len(marked_students)} students:")
            for student in marked_students:
                st.write(f"• {student}")
        else:
            st.info("📝 No new attendance marked")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Attendance System",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Smart Attendance System")
    st.markdown("**Step-by-step attendance marking with QR verification and face recognition**")
    
    # Initialize session state
    initialize_session_state()
    
    # Progress bar
    progress_steps = ["QR Code Scan", "Face Recognition"]
    progress = (st.session_state.step - 1) / len(progress_steps)
    st.progress(progress)
    
    # Step indicator
    step_cols = st.columns(len(progress_steps))
    for i, step_name in enumerate(progress_steps):
        with step_cols[i]:
            if i + 1 == st.session_state.step:
                st.markdown(f"**🔄 {i+1}. {step_name}**")
            elif i + 1 < st.session_state.step:
                st.markdown(f"✅ {i+1}. {step_name}")
            else:
                st.markdown(f"⏳ {i+1}. {step_name}")
    
    st.divider()
    
    # Execute current step
    if st.session_state.step == 1:
        step1_qr_scan()
    elif st.session_state.step == 2:
        step2_face_recognition()
    
    # Footer
    st.divider()
    st.markdown("---")
    st.caption("🤖 Smart Attendance System - Powered by Face Recognition & Firebase")

if __name__ == "__main__":
    main()
