"""
Simple Real-time Facial Recognition System
Uses MediaPipe landmarks directly without embedding model
Continuous recognition with 1-second print intervals
"""
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path
import time

from firebase_simple import SimpleFirebase

class SimpleFaceRecognizer:
    def __init__(self):
        """Initialize the facial recognition system"""
        self.setup_mediapipe()
        self.firebase = SimpleFirebase()
        self.students_cache = {}
        
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
            
    def cosine_similarity(self, landmarks1, landmarks2):
        """Compute cosine similarity between two landmark arrays"""
        # Normalize landmarks
        norm1 = np.linalg.norm(landmarks1)
        norm2 = np.linalg.norm(landmarks2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(landmarks1, landmarks2) / (norm1 * norm2)
        
    def euclidean_distance(self, landmarks1, landmarks2):
        """Compute euclidean distance between two landmark arrays"""
        return np.linalg.norm(landmarks1 - landmarks2)
        
    def register_student(self, class_id, student_id, student_name):
        """Register a new student using landmarks directly"""
        print(f"\n🎯 Registering: {student_name} ({student_id}) in {class_id}")
        print("📷 Starting camera...")
        
        cap = cv2.VideoCapture(0)
        landmark_sets = []
        
        print("📋 Instructions:")
        print("   • Look at camera, press SPACE to capture")
        print("   • Need 5+ captures, press 'q' when done")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Show frame
            cv2.putText(frame, f"Captures: {len(landmark_sets)}/5+", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: capture, Q: finish", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                landmarks = self.extract_landmarks(frame)
                if landmarks is not None:
                    landmark_sets.append(landmarks)
                    print(f"✅ Capture {len(landmark_sets)}")
                else:
                    print("❌ No face detected")
                    
            elif key == ord('q'):  # Quit
                if len(landmark_sets) >= 5:
                    break
                else:
                    print(f"Need at least 5 captures, got {len(landmark_sets)}")
                    
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)  # Brief pause
        
        if len(landmark_sets) >= 5:
            # Save to Firebase using existing method but with landmarks instead of embeddings
            success = self.firebase.register_student(class_id, student_id, student_name, landmark_sets)
            if success:
                print(f"✅ Registered {student_name} with {len(landmark_sets)} landmark sets")
            return success
        else:
            print("❌ Registration failed - insufficient captures")
            return False
            
    def load_class_students(self, class_id):
        """Load students and their landmarks for recognition"""
        try:
            students = {}
            students_ref = self.firebase.db.collection('classes').document(class_id).collection('students')
            
            for student_doc in students_ref.stream():
                student_data = student_doc.to_dict()
                if not student_data.get('is_active', True):
                    continue
                    
                student_id = student_doc.id
                
                # Get landmark sets (stored as embeddings in Firebase)
                landmark_sets = []
                embeddings_ref = student_doc.reference.collection('embeddings')
                for emb_doc in embeddings_ref.stream():
                    emb_data = emb_doc.to_dict()
                    landmark_sets.append(np.array(emb_data['embedding']))
                
                if landmark_sets:
                    students[student_id] = {
                        'name': student_data['name'],
                        'landmarks': landmark_sets
                    }
            
            self.students_cache = students
            print(f"✅ Loaded {len(self.students_cache)} students for recognition")
            return students
            
        except Exception as e:
            print(f"❌ Error loading students: {e}")
            self.students_cache = {}
            return {}
            
    def recognize_face(self, landmarks, threshold=0.8):
        """Recognize face from landmarks using multiple similarity metrics"""
        if not self.students_cache:
            return None, 0.0
            
        if landmarks is None:
            return None, 0.0
            
        best_match = None
        best_score = -1
        
        for student_id, student_data in self.students_cache.items():
            student_name = student_data['name']
            stored_landmarks = student_data['landmarks']
            
            # Calculate similarity with each stored landmark set
            max_similarity = -1
            for stored_landmark in stored_landmarks:
                # Use cosine similarity
                similarity = self.cosine_similarity(landmarks, stored_landmark)
                
                # Also use euclidean distance (inverted and normalized)
                distance = self.euclidean_distance(landmarks, stored_landmark)
                # Normalize distance to 0-1 range (roughly)
                distance_score = max(0, 1 - (distance / 100))  # Adjust divisor as needed
                
                # Combine both metrics
                combined_score = (similarity * 0.7) + (distance_score * 0.3)
                
                if combined_score > max_similarity:
                    max_similarity = combined_score
                    
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = (student_id, student_name)
                
        if best_score >= threshold:
            return best_match, best_score
        else:
            return None, best_score
            
    def live_recognition(self, class_id, threshold=0.8):
        """Live face recognition with continuous display and 1-second print intervals"""
        print(f"\n🎯 Starting live recognition for {class_id}")
        
        # Load students
        self.load_class_students(class_id)
        if not self.students_cache:
            print("❌ No students found in class")
            return
            
        print(f"🔍 Threshold: {threshold} | Press 'q' to quit, 'r' to reload students")
        print(f"📈 Higher threshold = More strict recognition")
        print(f"⏱️  Recognition results printed every 1 second")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera to 30 FPS if possible
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Camera FPS: {actual_fps}")
        
        # Variables for continuous recognition
        current_recognition = None
        current_confidence = 0.0
        last_print_time = 0
        print_interval = 1.0  # Print every 1 second
        
        # Statistics
        frame_count = 0
        recognition_count = 0
        successful_recognition = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time()
            frame_count += 1
            
            # Extract landmarks and recognize face every frame
            landmarks = self.extract_landmarks(frame)
            if landmarks is not None:
                match, confidence = self.recognize_face(landmarks, threshold)
                recognition_count += 1
                
                if match:
                    student_id, student_name = match
                    current_recognition = student_name
                    current_confidence = confidence
                    successful_recognition += 1
                else:
                    current_recognition = "Unknown"
                    current_confidence = confidence
            else:
                current_recognition = "No face detected"
                current_confidence = 0.0
            
            # Print recognition result every second
            if current_time - last_print_time >= print_interval:
                if current_recognition:
                    if current_recognition != "No face detected" and current_recognition != "Unknown":
                        print(f"🔍 RECOGNIZED: {current_recognition} (confidence: {current_confidence:.3f})")
                    elif current_recognition == "Unknown":
                        print(f"❓ Unknown face detected (best score: {current_confidence:.3f})")
                    else:
                        print(f"👁️  {current_recognition}")
                
                last_print_time = current_time
            
            # Draw landmarks on frame
            if landmarks is not None:
                self._draw_face_landmarks(frame, landmarks)
            
            # Display current recognition on frame
            self._display_overlay(frame, current_recognition, current_confidence, threshold, frame_count, recognition_count, successful_recognition, start_time, current_time)
            
            cv2.imshow('Live Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reload students
                print("🔄 Reloading students...")
                self.load_class_students(class_id)
                print(f"✅ Reloaded {len(self.students_cache)} students")
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        success_rate = (successful_recognition / recognition_count * 100) if recognition_count > 0 else 0
        
        print(f"\n📊 Session Summary:")
        print(f"   Duration: {total_time:.1f}s")
        print(f"   Total frames: {frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Recognition attempts: {recognition_count}")
        print(f"   Successful recognitions: {successful_recognition}")
        print(f"   Success rate: {success_rate:.1f}%")
        
    def _draw_face_landmarks(self, frame, landmarks):
        """Draw face landmarks on the frame"""
        h, w, _ = frame.shape
        
        # Convert normalized landmarks to pixel coordinates
        points = []
        for i in range(0, len(landmarks), 3):
            x = int(landmarks[i] * w)
            y = int(landmarks[i + 1] * h)
            points.append((x, y))
        
        # Draw key facial landmarks
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
        
        # Draw eyes
        for indices, color in [(left_eye_indices, (255, 0, 0)), (right_eye_indices, (255, 0, 0))]:
            for i in range(len(indices)):
                if indices[i] < len(points):
                    pt1 = points[indices[i]]
                    pt2 = points[indices[(i + 1) % len(indices)]]
                    cv2.line(frame, pt1, pt2, color, 1)
        
        # Lips outline
        outer_lip_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 267, 271, 272, 271, 272]
        for i in range(len(outer_lip_indices)):
            if outer_lip_indices[i] < len(points):
                pt1 = points[outer_lip_indices[i]]
                pt2 = points[outer_lip_indices[(i + 1) % len(outer_lip_indices)]]
                cv2.line(frame, pt1, pt2, (0, 0, 255), 1)  # Red for lips
        
    def _display_overlay(self, frame, current_recognition, current_confidence, threshold, frame_count, recognition_count, successful_recognition, start_time, current_time):
        """Display overlay information on frame"""
        # Create semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Show current recognition
        if current_recognition:
            if current_recognition not in ["No face detected", "Unknown"]:
                color = (0, 255, 0)  # Green for recognized
                text = f"RECOGNIZED: {current_recognition}"
            elif current_recognition == "Unknown":
                color = (0, 165, 255)  # Orange for unknown
                text = f"Unknown Face"
            else:
                color = (128, 128, 128)  # Gray for no face
                text = current_recognition
                
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if current_confidence > 0:
                cv2.putText(frame, f"Confidence: {current_confidence:.3f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show statistics
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.1f} | Frames: {frame_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if recognition_count > 0:
            success_rate = (successful_recognition / recognition_count) * 100
            cv2.putText(frame, f"Success: {success_rate:.1f}% ({successful_recognition}/{recognition_count})", 
                       (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show threshold
        cv2.putText(frame, f"Threshold: {threshold:.2f}", 
                   (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'r' to reload students", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def main():
    parser = argparse.ArgumentParser(description='Simple Real-time Face Recognition')
    parser.add_argument('--mode', choices=['register', 'live'], required=True,
                       help='Mode: register new student or live recognition')
    parser.add_argument('--class-id', required=True,
                       help='Class identifier')
    parser.add_argument('--student-id', help='Student ID (for registration mode)')
    parser.add_argument('--student-name', help='Student name (for registration mode)')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Recognition threshold (0.0-1.0, higher = more strict)')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("❌ Threshold must be between 0.0 and 1.0")
        return
    
    system = SimpleFaceRecognizer()
    
    if args.mode == 'register':
        if not args.student_id or not args.student_name:
            print("❌ For registration mode, need --student-id and --student-name")
            print("Example: python facerec.py --mode register --class-id CS101 --student-id S001 --student-name 'John Doe'")
            return
            
        system.register_student(args.class_id, args.student_id, args.student_name)
        
    elif args.mode == 'live':
        system.live_recognition(args.class_id, args.threshold)

if __name__ == "__main__":
    main()
