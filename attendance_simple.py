"""
Simple Face Recognition Attendance System
Single file for registration and live attendance tracking
"""
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import argparse
from pathlib import Path
import time

from firebase_simple import SimpleFirebase

class SimpleEmbeddingModel(nn.Module):
    """Improved embedding model with better architecture"""
    def __init__(self, input_dim=1434, embedding_dim=128):
        super().__init__()
        
        # Add input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Deeper network with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, embedding_dim)
        )
        
        # Final normalization
        self.final_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        # Handle both single samples and batches
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Normalize input
        x = self.input_norm(x)
        
        # Generate embedding
        embedding = self.encoder(x)
        
        # Final normalization
        embedding = self.final_norm(embedding)
        
        # L2 normalize for cosine similarity
        return F.normalize(embedding, p=2, dim=1)

class FaceRecognitionSystem:
    def __init__(self):
        """Initialize the face recognition system"""
        self.setup_mediapipe()
        self.load_model()
        self.firebase = SimpleFirebase()
        self.students_cache = {}
        self.frame_count = 0
        
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
        """Load trained embedding model - try improved model first"""
        # Try improved model first
        improved_path = Path("trained_models/improved_face_model.pth")
        simple_path = Path("trained_models/simple_face_model.pth")
        
        if improved_path.exists():
            print("🔄 Loading improved model...")
            self.model = SimpleEmbeddingModel()
            try:
                self.model.load_state_dict(torch.load(improved_path))
                self.model.eval()
                print("✅ Improved model loaded")
                return
            except Exception as e:
                print(f"⚠️ Failed to load improved model: {e}")
        
        if simple_path.exists():
            print("🔄 Loading simple model...")
            self.model = SimpleEmbeddingModel()
            try:
                self.model.load_state_dict(torch.load(simple_path))
                self.model.eval()
                print("✅ Simple model loaded")
                return
            except Exception as e:
                print(f"❌ Failed to load simple model: {e}")
        
        print("❌ No trained model found! Run retrain_improved.py first")
        self.model = None
        
    def extract_landmarks(self, image):
        """Extract face landmarks from image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
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
        # Embeddings are already normalized by the model
        return np.dot(emb1, emb2)
        
    def register_student(self, class_id, student_id, student_name):
        """Register a new student"""
        print(f"\n🎯 Registering: {student_name} ({student_id}) in {class_id}")
        print("📷 Starting camera...")
        
        cap = cv2.VideoCapture(0)
        embeddings = []
        
        print("📋 Instructions:")
        print("   • Look at camera, press SPACE to capture")
        print("   • Need 5+ captures, press 'q' when done")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Show frame
            cv2.putText(frame, f"Captures: {len(embeddings)}/5+", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: capture, Q: finish", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                landmarks = self.extract_landmarks(frame)
                if landmarks is not None:
                    embedding = self.generate_embedding(landmarks)
                    if embedding is not None:
                        embeddings.append(embedding)
                        print(f"✅ Capture {len(embeddings)}")
                    else:
                        print("❌ Failed to generate embedding")
                else:
                    print("❌ No face detected")
                    
            elif key == ord('q'):  # Quit
                if len(embeddings) >= 5:
                    break
                else:
                    print(f"Need at least 5 captures, got {len(embeddings)}")
                    
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)  # Brief pause
        
        if len(embeddings) >= 5:
            # Save to Firebase
            success = self.firebase.register_student(class_id, student_id, student_name, embeddings)
            if success:
                print(f"✅ Registered {student_name} with {len(embeddings)} embeddings")
            return success
        else:
            print("❌ Registration failed - insufficient captures")
            return False
            
    def load_class_students(self, class_id):
        """Load students for recognition"""
        self.students_cache = self.firebase.get_class_students(class_id)
        print(f"✅ Loaded {len(self.students_cache)} students for recognition")
        
    def recognize_face(self, landmarks, threshold=0.7):
        """Recognize face from landmarks"""
        if not self.students_cache:
            return None, 0.0
            
        query_embedding = self.generate_embedding(landmarks)
        if query_embedding is None:
            return None, 0.0
            
        best_match = None
        best_similarity = -1
        
        for student_id, student_data in self.students_cache.items():
            for stored_embedding in student_data['embeddings']:
                similarity = self.cosine_similarity(query_embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (student_id, student_data['name'])
                    
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity
            
    def live_recognition(self, class_id, threshold=0.85):  # Increased default threshold for safety
        """Live face recognition and attendance with multi-frame verification"""
        print(f"\n🎯 Starting live recognition for {class_id}")
        
        # Load students
        self.load_class_students(class_id)
        if not self.students_cache:
            print("❌ No students found in class")
            return
            
        print(f"🔍 Threshold: {threshold} | Press 'q' to quit, 'r' to reload students")
        print(f"📈 Higher threshold = More strict recognition")
        print(f"🎯 Multi-frame verification: 5 consecutive detections (0.25s apart) required")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera to 30 FPS if possible
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Camera FPS: {actual_fps}")
        
        recognition_count = 0
        successful_recognition = 0
        last_recognized_student = None
        last_recognition_time = 0
        recognition_display_time = 0  # Time to display recognition result
        last_attendance_marked = {}  # Track last attendance marking per student
        
        # Multi-frame verification variables
        verification_frames = []  # List of (student_id, student_name, confidence, timestamp)
        required_frames = 5
        frame_interval = 0.25  # 0.25 seconds between verification frames
        last_verification_time = 0
        pending_verification = {}  # Track verification progress per student
        verification_window = 2.0  # Total time window for verification (5 frames * 0.25s = 1.25s + buffer)
        
        # Variables for 1-second recognition interval
        last_process_time = 0
        current_landmarks = None  # Store current frame landmarks for display
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time()
            
            # Always extract landmarks for display (lightweight operation)
            landmarks = self.extract_landmarks(frame)
            if landmarks is not None:
                current_landmarks = landmarks
            
            # Multi-frame verification process (every 0.25 seconds)
            if current_time - last_verification_time >= frame_interval:
                last_verification_time = current_time
                
                if landmarks is not None:
                    # Recognize face
                    match, confidence = self.recognize_face(landmarks, threshold)
                    recognition_count += 1
                    
                    if match:
                        student_id, student_name = match
                        successful_recognition += 1
                        
                        # Add to verification frames
                        verification_frames.append({
                            'student_id': student_id,
                            'student_name': student_name,
                            'confidence': confidence,
                            'timestamp': current_time
                        })
                        
                        # Update pending verification for this student
                        if student_id not in pending_verification:
                            pending_verification[student_id] = []
                        pending_verification[student_id].append({
                            'confidence': confidence,
                            'timestamp': current_time
                        })
                        
                        # Show immediate recognition feedback
                        last_recognized_student = (student_name, confidence)
                        last_recognition_time = current_time
                        recognition_display_time = current_time + 2.0
                        
                        print(f"🔍 Detection {len(pending_verification[student_id])}/5: {student_name} (confidence: {confidence:.3f})")
                
                # Clean old verification frames (outside time window)
                current_verification_frames = []
                for frame_data in verification_frames:
                    if current_time - frame_data['timestamp'] <= verification_window:
                        current_verification_frames.append(frame_data)
                verification_frames = current_verification_frames
                
                # Clean old pending verifications
                for student_id in list(pending_verification.keys()):
                    valid_detections = []
                    for detection in pending_verification[student_id]:
                        if current_time - detection['timestamp'] <= verification_window:
                            valid_detections.append(detection)
                    
                    if valid_detections:
                        pending_verification[student_id] = valid_detections
                    else:
                        del pending_verification[student_id]
                
                # Check for completed verifications
                for student_id in list(pending_verification.keys()):
                    detections = pending_verification[student_id]
                    if len(detections) >= required_frames:
                        # Calculate average confidence
                        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                        student_name = next((d['student_name'] for d in verification_frames if d['student_id'] == student_id), "Unknown")
                        
                        # Check if already marked recently
                        recent_check = self.firebase.check_recent_attendance(class_id, student_id, minutes=5)
                        
                        if not recent_check:
                            try:
                                if self.firebase.mark_attendance(class_id, student_id, avg_confidence):
                                    print(f"✅ {student_name} - ATTENDANCE MARKED! (avg confidence: {avg_confidence:.3f}, {len(detections)} detections)")
                                    last_attendance_marked[student_id] = current_time
                                    
                                    # Clear this student's verification data
                                    if student_id in pending_verification:
                                        del pending_verification[student_id]
                                    
                                    # Update display
                                    last_recognized_student = (f"✅ {student_name} - MARKED", avg_confidence)
                                    recognition_display_time = current_time + 3.0  # Show longer for attendance marking
                                else:
                                    print(f"❌ Failed to mark attendance for {student_name}")
                            except Exception as e:
                                print(f"❌ Error marking attendance for {student_name}: {e}")
                        else:
                            print(f"⏭️  {student_name} - Already marked recently, skipping")
                            if student_id in pending_verification:
                                del pending_verification[student_id]
            
            # Draw face landmarks if available
            if current_landmarks is not None:
                self._draw_face_landmarks(frame, current_landmarks)
            
            # Display overlay with verification progress
            self._display_overlay(frame, last_recognized_student, recognition_display_time, current_time, recognition_count, successful_recognition, threshold, pending_verification)
            cv2.imshow('Live Recognition', frame)
            
            # Use waitKey(33) for ~30 FPS (1000ms/30 ≈ 33ms)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reload students
                print("🔄 Reloading students...")
                self.load_class_students(class_id)
                print(f"✅ Reloaded {len(self.students_cache)} students")
                
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Session Summary:")
        print(f"   Total recognitions: {recognition_count}")
        print(f"   Successful: {successful_recognition}")
        print(f"   Success rate: {(successful_recognition/recognition_count)*100:.1f}%" if recognition_count > 0 else "   Success rate: 0%")
        
    def _draw_face_landmarks(self, frame, landmarks):
        """Draw face landmarks on the frame"""
        h, w, _ = frame.shape
        
        # Convert normalized landmarks to pixel coordinates
        # landmarks array has 478 points * 3 coordinates = 1434 values
        points = []
        for i in range(0, len(landmarks), 3):
            x = int(landmarks[i] * w)
            y = int(landmarks[i + 1] * h)
            points.append((x, y))
        
        # Draw key facial landmarks with different colors
        # Face oval (outline)
        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        for i in range(len(face_oval_indices)):
            if i < len(points) and face_oval_indices[i] < len(points):
                pt1 = points[face_oval_indices[i]]
                pt2 = points[face_oval_indices[(i + 1) % len(face_oval_indices)]]
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1)  # Yellow outline
        
        # Eyes
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Draw left eye
        for i in range(len(left_eye_indices)):
            if left_eye_indices[i] < len(points):
                pt1 = points[left_eye_indices[i]]
                pt2 = points[left_eye_indices[(i + 1) % len(left_eye_indices)]]
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)  # Blue for left eye
        
        # Draw right eye  
        for i in range(len(right_eye_indices)):
            if right_eye_indices[i] < len(points):
                pt1 = points[right_eye_indices[i]]
                pt2 = points[right_eye_indices[(i + 1) % len(right_eye_indices)]]
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)  # Blue for right eye
        
        # Eyebrows
        left_eyebrow_indices = [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305]
        right_eyebrow_indices = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276]
        
        for i in range(len(left_eyebrow_indices) - 1):
            if left_eyebrow_indices[i] < len(points) and left_eyebrow_indices[i + 1] < len(points):
                pt1 = points[left_eyebrow_indices[i]]
                pt2 = points[left_eyebrow_indices[i + 1]]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)  # Green for eyebrows
        
        for i in range(len(right_eyebrow_indices) - 1):
            if right_eyebrow_indices[i] < len(points) and right_eyebrow_indices[i + 1] < len(points):
                pt1 = points[right_eyebrow_indices[i]]
                pt2 = points[right_eyebrow_indices[i + 1]]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)  # Green for eyebrows
        
        # Nose
        nose_indices = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 328, 327, 326, 2]
        for i in range(len(nose_indices) - 1):
            if nose_indices[i] < len(points) and nose_indices[i + 1] < len(points):
                pt1 = points[nose_indices[i]]
                pt2 = points[nose_indices[i + 1]]
                cv2.line(frame, pt1, pt2, (255, 255, 0), 1)  # Cyan for nose
        
        # Lips
        outer_lip_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 267, 271, 272, 271, 272]
        inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78]
        
        for i in range(len(outer_lip_indices)):
            if outer_lip_indices[i] < len(points):
                pt1 = points[outer_lip_indices[i]]
                pt2 = points[outer_lip_indices[(i + 1) % len(outer_lip_indices)]]
                cv2.line(frame, pt1, pt2, (0, 0, 255), 1)  # Red for lips
        
        # Draw key landmark points
        key_points = [10, 151, 9, 175, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        for idx in key_points:
            if idx < len(points):
                cv2.circle(frame, points[idx], 2, (255, 255, 255), -1)  # White dots for key points
        
    def _display_overlay(self, frame, last_recognized_student, recognition_display_time, current_time, recognition_count, successful_recognition, threshold=0.7, pending_verification=None):
        """Helper method to display overlay text consistently"""
        # Clear area for text (add a semi-transparent background)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Show last recognized student if within display time
        if last_recognized_student and current_time < recognition_display_time:
            student_name, confidence = last_recognized_student
            cv2.putText(frame, f"RECOGNIZED: {student_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show verification progress
        if pending_verification:
            y_offset = 90
            for student_id, detections in pending_verification.items():
                progress = len(detections)
                student_name = "Verifying..."
                if detections:
                    # Get the most recent student name from detections (assuming we have it)
                    pass
                
                # Show progress bar
                progress_text = f"Verifying: {progress}/5 frames"
                cv2.putText(frame, progress_text, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Progress bar visualization
                bar_width = 200
                bar_height = 10
                bar_x = 250
                bar_y = y_offset - 10
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                
                # Progress bar
                progress_width = int((progress / 5) * bar_width)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 255), -1)
                
                y_offset += 25
                break  # Only show one verification at a time to avoid clutter
        
        # Show stats in bottom area
        if recognition_count > 0:
            success_rate = (successful_recognition / recognition_count) * 100
            cv2.putText(frame, f"Success: {success_rate:.1f}% ({successful_recognition}/{recognition_count})", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show FPS and instructions
        cv2.putText(frame, "Press 'q' to quit | 'r' to reload | Multi-frame verification: ON", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show threshold info
        cv2.putText(frame, f"Threshold: {threshold:.2f}", 
                   (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    parser = argparse.ArgumentParser(description='Simple Face Recognition Attendance')
    parser.add_argument('--mode', choices=['register', 'live'], required=True)
    parser.add_argument('--class-id', required=True)
    parser.add_argument('--student-id', help='For registration mode')
    parser.add_argument('--student-name', help='For registration mode')
    parser.add_argument('--threshold', type=float, default=0.85, help='Recognition threshold (higher = more strict, RECOMMENDED: 0.85+)')
    
    args = parser.parse_args()
    
    system = FaceRecognitionSystem()
    
    if args.mode == 'register':
        if not args.student_id or not args.student_name:
            print("❌ For registration mode, need --student-id and --student-name")
            return
            
        system.register_student(args.class_id, args.student_id, args.student_name)
        
    elif args.mode == 'live':
        system.live_recognition(args.class_id, args.threshold)

if __name__ == "__main__":
    main()
