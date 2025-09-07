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
        """Load trained embedding model"""
        model_path = Path("trained_models/simple_face_model.pth")
        
        if not model_path.exists():
            print("❌ No trained model found! Run train_simple.py first")
            self.model = None
            return
            
        self.model = SimpleEmbeddingModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("✅ Model loaded")
        
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
            
    def live_recognition(self, class_id, threshold=0.7):
        """Live face recognition and attendance"""
        print(f"\n🎯 Starting live recognition for {class_id}")
        
        # Load students
        self.load_class_students(class_id)
        if not self.students_cache:
            print("❌ No students found in class")
            return
            
        print(f"🔍 Threshold: {threshold} | Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        recognition_count = 0
        successful_recognition = 0
        skip_frames = 0  # Skip some frames for better performance
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 3rd frame for better performance
            skip_frames += 1
            if skip_frames % 3 != 0:
                cv2.imshow('Live Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            
            if landmarks is not None:
                # Recognize face
                match, confidence = self.recognize_face(landmarks, threshold)
                recognition_count += 1
                
                if match:
                    student_id, student_name = match
                    successful_recognition += 1
                    
                    # Check if already marked recently
                    if not self.firebase.check_recent_attendance(class_id, student_id):
                        # Mark attendance
                        if self.firebase.mark_attendance(class_id, student_id, confidence):
                            print(f"✅ {student_name} - Attendance marked (confidence: {confidence:.3f})")
                        
                    # Show recognition on frame
                    cv2.putText(frame, f"{student_name}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{confidence:.3f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Show no match (only display every 30 frames to reduce lag)
                    if recognition_count % 30 == 0:
                        cv2.putText(frame, f"No match", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Only show "no face" message every 60 frames to reduce clutter
                if recognition_count % 60 == 0:
                    cv2.putText(frame, "No face detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
            # Show stats
            if recognition_count > 0:
                success_rate = (successful_recognition / recognition_count) * 100
                cv2.putText(frame, f"Success: {success_rate:.1f}% ({successful_recognition}/{recognition_count})", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            cv2.imshow('Live Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Session Summary:")
        print(f"   Total recognitions: {recognition_count}")
        print(f"   Successful: {successful_recognition}")
        print(f"   Success rate: {(successful_recognition/recognition_count)*100:.1f}%" if recognition_count > 0 else "   Success rate: 0%")

def main():
    parser = argparse.ArgumentParser(description='Simple Face Recognition Attendance')
    parser.add_argument('--mode', choices=['register', 'live'], required=True)
    parser.add_argument('--class-id', required=True)
    parser.add_argument('--student-id', help='For registration mode')
    parser.add_argument('--student-name', help='For registration mode')
    parser.add_argument('--threshold', type=float, default=0.7, help='Recognition threshold')
    
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
