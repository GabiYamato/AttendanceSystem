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
        
        # Key landmark indices for facial features (reduced from 468 to ~68 key points)
        self.key_landmarks = {
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49],
            'right_eyebrow': [276, 283, 282, 295, 285, 336, 296, 334, 293, 300],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 328, 327, 326],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 271, 272]
        }
        
        # Flatten all key landmarks into a single list (remove duplicates)
        all_key_indices = set()
        for feature_indices in self.key_landmarks.values():
            all_key_indices.update(feature_indices)
        self.key_landmark_indices = sorted(list(all_key_indices))
        
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
        """Extract and normalize face landmarks from image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            
            # Detect face first
            detection_results = self.face_detection.process(rgb_image)
            if not detection_results.detections:
                return None, None
                
            # Get face bounding box for quality assessment
            detection = detection_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            face_area = bbox.width * bbox.height
            
            # Quality check: face should be reasonable size
            if face_area < 0.1:  # Face too small
                return None, "Face too small - move closer"
            if face_area > 0.8:  # Face too large
                return None, "Face too large - move back"
                
            # Extract landmarks
            results = self.face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return None, None
                
            # Get all landmarks
            landmarks = results.multi_face_landmarks[0]
            
            # Extract key landmark coordinates
            key_points = []
            all_points = []
            
            for i, landmark in enumerate(landmarks.landmark):
                point = [landmark.x, landmark.y, landmark.z]
                all_points.append(point)
                if i in self.key_landmark_indices:
                    key_points.append(point)
            
            # Convert to numpy arrays
            key_points = np.array(key_points)
            all_points = np.array(all_points)
            
            # Normalize landmarks
            normalized_key = self.normalize_landmarks(key_points)
            if normalized_key is None:
                return None, "Failed to normalize landmarks"
                
            # Calculate geometric ratios for additional features
            geometric_features = self.calculate_geometric_features(all_points)
            
            # Combine normalized landmarks with geometric features
            combined_features = np.concatenate([normalized_key.flatten(), geometric_features])
            
            # Quality assessment
            quality_score = self.assess_landmark_quality(all_points, bbox)
            
            return combined_features, quality_score
            
        except Exception as e:
            return None, f"Error: {str(e)}"
            
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks for pose and scale invariance"""
        try:
            # Method 1: Normalize by bounding box
            min_coords = np.min(landmarks[:, :2], axis=0)  # Only x, y
            max_coords = np.max(landmarks[:, :2], axis=0)
            
            # Avoid division by zero
            ranges = max_coords - min_coords
            if np.any(ranges == 0):
                return None
                
            # Normalize x, y coordinates to [0, 1] within face bounding box
            normalized = landmarks.copy()
            normalized[:, :2] = (landmarks[:, :2] - min_coords) / ranges
            
            # Method 2: Additional normalization by inter-ocular distance
            # Find eye landmarks (approximate indices in key landmarks)
            try:
                left_eye_center = np.mean(landmarks[6:22, :2], axis=0)  # Rough eye region
                right_eye_center = np.mean(landmarks[22:38, :2], axis=0)  # Rough eye region
                inter_ocular_distance = np.linalg.norm(right_eye_center - left_eye_center)
                
                if inter_ocular_distance > 0:
                    # Scale by inter-ocular distance
                    normalized[:, :2] = normalized[:, :2] / inter_ocular_distance
            except:
                pass  # Fallback to bounding box normalization only
                
            # Method 3: Center around face centroid
            centroid = np.mean(normalized[:, :2], axis=0)
            normalized[:, :2] = normalized[:, :2] - centroid
            
            return normalized
            
        except Exception as e:
            return None
            
    def calculate_geometric_features(self, all_landmarks):
        """Calculate geometric ratios and features that are pose-invariant"""
        try:
            features = []
            
            # Key point indices (approximate - adjust based on MediaPipe documentation)
            nose_tip = 1
            left_eye_outer = 33
            right_eye_outer = 362
            left_mouth = 61
            right_mouth = 291
            chin = 175
            
            # Ensure indices are valid
            max_idx = len(all_landmarks) - 1
            if any(idx > max_idx for idx in [nose_tip, left_eye_outer, right_eye_outer, left_mouth, right_mouth, chin]):
                # Fallback to basic features
                return np.array([0.0] * 20)  # Return zeros if landmark extraction fails
            
            # Distance ratios (pose invariant)
            eye_distance = np.linalg.norm(all_landmarks[left_eye_outer][:2] - all_landmarks[right_eye_outer][:2])
            nose_to_chin = np.linalg.norm(all_landmarks[nose_tip][:2] - all_landmarks[chin][:2])
            mouth_width = np.linalg.norm(all_landmarks[left_mouth][:2] - all_landmarks[right_mouth][:2])
            
            if eye_distance > 0:
                features.extend([
                    nose_to_chin / eye_distance,  # Face length ratio
                    mouth_width / eye_distance,   # Mouth width ratio
                ])
            else:
                features.extend([1.0, 0.5])  # Default ratios
                
            # Angle features
            # Face symmetry: angles between key points
            try:
                # Vector from nose to left eye
                nose_to_left_eye = all_landmarks[left_eye_outer][:2] - all_landmarks[nose_tip][:2]
                # Vector from nose to right eye
                nose_to_right_eye = all_landmarks[right_eye_outer][:2] - all_landmarks[nose_tip][:2]
                
                # Calculate angle between vectors
                dot_product = np.dot(nose_to_left_eye, nose_to_right_eye)
                norms = np.linalg.norm(nose_to_left_eye) * np.linalg.norm(nose_to_right_eye)
                if norms > 0:
                    angle = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
                    features.append(angle)
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
                
            # Add more geometric features (triangular areas, etc.)
            # Triangle area ratios
            try:
                # Eye-nose triangle area
                eye_nose_area = self.triangle_area(
                    all_landmarks[left_eye_outer][:2],
                    all_landmarks[right_eye_outer][:2],
                    all_landmarks[nose_tip][:2]
                )
                
                # Nose-mouth-chin triangle area
                nose_mouth_area = self.triangle_area(
                    all_landmarks[nose_tip][:2],
                    all_landmarks[left_mouth][:2],
                    all_landmarks[chin][:2]
                )
                
                if eye_nose_area > 0:
                    features.append(nose_mouth_area / eye_nose_area)
                else:
                    features.append(1.0)
                    
            except:
                features.append(1.0)
                
            # Pad features to fixed length
            target_length = 20
            while len(features) < target_length:
                features.append(0.0)
                
            return np.array(features[:target_length])
            
        except Exception as e:
            # Return default feature vector if calculation fails
            return np.array([0.0] * 20)
            
    def triangle_area(self, p1, p2, p3):
        """Calculate area of triangle formed by three points"""
        return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))/2.0)
        
    def assess_landmark_quality(self, landmarks, bbox):
        """Assess quality of landmark detection"""
        try:
            quality_score = 1.0
            
            # Factor 1: Face size (optimal range)
            face_area = bbox.width * bbox.height
            if face_area < 0.15:
                quality_score *= 0.7  # Too small
            elif face_area > 0.6:
                quality_score *= 0.8  # Too large
                
            # Factor 2: Landmark spread (how well distributed they are)
            coords = landmarks[:, :2]
            std_x = np.std(coords[:, 0])
            std_y = np.std(coords[:, 1])
            
            if std_x < 0.05 or std_y < 0.05:  # Too clustered
                quality_score *= 0.6
                
            # Factor 3: Symmetry check (basic)
            try:
                left_points = coords[coords[:, 0] < 0.5]  # Points on left side
                right_points = coords[coords[:, 0] > 0.5]  # Points on right side
                
                if len(left_points) > 0 and len(right_points) > 0:
                    left_spread = np.std(left_points)
                    right_spread = np.std(right_points)
                    symmetry = 1.0 - abs(left_spread - right_spread)
                    quality_score *= max(0.5, symmetry)
            except:
                pass
                
            return max(0.0, min(1.0, quality_score))  # Clamp between 0 and 1
            
        except:
            return 0.5  # Default quality score
            
    def cosine_similarity(self, features1, features2):
        """Compute cosine similarity between two feature arrays"""
        # Normalize features
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(features1, features2) / (norm1 * norm2)
        
    def euclidean_distance(self, features1, features2):
        """Compute normalized euclidean distance between two feature arrays"""
        distance = np.linalg.norm(features1 - features2)
        # Normalize distance to 0-1 range (higher is more similar)
        max_possible_distance = np.sqrt(len(features1))  # Maximum possible distance
        normalized_distance = distance / max_possible_distance
        return max(0.0, 1.0 - normalized_distance)  # Convert to similarity score
        
    def correlation_similarity(self, features1, features2):
        """Compute correlation coefficient as similarity measure"""
        try:
            correlation = np.corrcoef(features1, features2)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return abs(correlation)  # Use absolute value
        except:
            return 0.0
        
    def register_student(self, class_id, student_id, student_name):
        """Register a new student using landmarks directly with quality control"""
        print(f"\n🎯 Registering: {student_name} ({student_id}) in {class_id}")
        print("📷 Starting camera...")
        
        cap = cv2.VideoCapture(0)
        feature_sets = []
        quality_scores = []
        
        print("📋 Instructions:")
        print("   • Position your face in the alignment box")
        print("   • Look straight at camera, press SPACE to capture")
        print("   • Need 5+ high-quality captures, press 'q' when done")
        print("   • Move slightly between captures for better coverage")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw alignment box and guidelines
            frame_with_guide = self.draw_alignment_guide(frame)
            
            # Extract landmarks and assess quality
            features, quality = self.extract_landmarks(frame)
            
            # Handle quality being a string (error message) or float
            if isinstance(quality, str):
                quality_text = quality
                status_color = (0, 0, 255)  # Red for error
                quality_score = 0.0
            elif quality is not None:
                quality_score = float(quality)
                status_color = (0, 255, 0) if quality_score > 0.7 else (0, 165, 255) if quality_score > 0.5 else (0, 0, 255)
                quality_text = f"Quality: {quality_score:.2f}"
                if quality_score > 0.8:
                    quality_text += " (EXCELLENT - Press SPACE)"
                elif quality_score > 0.6:
                    quality_text += " (GOOD - Press SPACE)"
                else:
                    quality_text += " (POOR - Adjust position)"
            else:
                quality_text = "No face detected"
                status_color = (128, 128, 128)
                quality_score = 0.0
                
            cv2.putText(frame_with_guide, quality_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame_with_guide, f"Captures: {len(feature_sets)}/5+ (avg quality: {np.mean(quality_scores) if quality_scores else 0:.2f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_guide, "SPACE: capture, Q: finish", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Registration', frame_with_guide)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                if features is not None and isinstance(quality, (int, float)) and quality > 0.5:  # Minimum quality threshold
                    feature_sets.append(features)
                    quality_scores.append(quality)
                    print(f"✅ Capture {len(feature_sets)} - Quality: {quality:.3f}")
                else:
                    error_msg = quality if isinstance(quality, str) else f"Quality: {quality if quality else 0:.3f}"
                    print(f"❌ Poor quality capture - {error_msg}")
                    
            elif key == ord('q'):  # Quit
                if len(feature_sets) >= 5:
                    # Check average quality
                    avg_quality = np.mean(quality_scores)
                    if avg_quality >= 0.6:
                        break
                    else:
                        print(f"⚠️  Average quality too low ({avg_quality:.3f}). Need better captures.")
                        continue
                else:
                    print(f"Need at least 5 captures, got {len(feature_sets)}")
                    
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)  # Brief pause
        
        if len(feature_sets) >= 5:
            # Quality filter: keep only the best captures if we have more than needed
            if len(feature_sets) > 8:
                # Sort by quality and keep top 8
                sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
                feature_sets = [feature_sets[i] for i in sorted_indices[:8]]
                quality_scores = [quality_scores[i] for i in sorted_indices[:8]]
                print(f"📊 Filtered to top 8 captures (quality range: {min(quality_scores):.3f} - {max(quality_scores):.3f})")
            
            # Save to Firebase
            success = self.firebase.register_student(class_id, student_id, student_name, feature_sets)
            if success:
                avg_quality = np.mean(quality_scores)
                print(f"✅ Registered {student_name} with {len(feature_sets)} feature sets (avg quality: {avg_quality:.3f})")
            return success
        else:
            print("❌ Registration failed - insufficient quality captures")
            return False
            
    def draw_alignment_guide(self, frame):
        """Draw face alignment guide on frame"""
        h, w, _ = frame.shape
        frame_copy = frame.copy()
        
        # Calculate optimal face box (roughly 40% of frame width, centered)
        box_width = int(w * 0.4)
        box_height = int(box_width * 1.3)  # Face is taller than wide
        
        center_x, center_y = w // 2, h // 2
        box_x1 = center_x - box_width // 2
        box_y1 = center_y - box_height // 2 - 20  # Slightly higher
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height
        
        # Draw outer guide box
        cv2.rectangle(frame_copy, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
        
        # Draw corner markers
        corner_size = 20
        corners = [(box_x1, box_y1), (box_x2, box_y1), (box_x1, box_y2), (box_x2, box_y2)]
        for x, y in corners:
            cv2.line(frame_copy, (x-corner_size, y), (x+corner_size, y), (0, 255, 255), 3)
            cv2.line(frame_copy, (x, y-corner_size), (x, y+corner_size), (0, 255, 255), 3)
        
        # Draw center cross for nose alignment
        cross_size = 15
        cv2.line(frame_copy, (center_x-cross_size, center_y), (center_x+cross_size, center_y), (255, 255, 255), 2)
        cv2.line(frame_copy, (center_x, center_y-cross_size), (center_x, center_y+cross_size), (255, 255, 255), 2)
        
        # Draw eye level guide
        eye_y = center_y - box_height // 4
        cv2.line(frame_copy, (box_x1, eye_y), (box_x2, eye_y), (0, 255, 0), 1)
        
        # Instructions overlay
        cv2.putText(frame_copy, "Align face with yellow box", (10, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_copy, "Center nose on white cross", (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy
            
    def load_class_students(self, class_id):
        """Load students and their feature sets for recognition"""
        try:
            students = {}
            students_ref = self.firebase.db.collection('classes').document(class_id).collection('students')
            
            for student_doc in students_ref.stream():
                student_data = student_doc.to_dict()
                if not student_data.get('is_active', True):
                    continue
                    
                student_id = student_doc.id
                
                # Get feature sets (stored as embeddings in Firebase)
                feature_sets = []
                embeddings_ref = student_doc.reference.collection('embeddings')
                for emb_doc in embeddings_ref.stream():
                    emb_data = emb_doc.to_dict()
                    feature_sets.append(np.array(emb_data['embedding']))
                
                if feature_sets:
                    students[student_id] = {
                        'name': student_data['name'],
                        'features': feature_sets
                    }
            
            self.students_cache = students
            print(f"✅ Loaded {len(self.students_cache)} students for recognition")
            return students
            
        except Exception as e:
            print(f"❌ Error loading students: {e}")
            self.students_cache = {}
            return {}
            
    def recognize_face(self, features, threshold=0.7):
        """Recognize face from features using multiple similarity metrics"""
        if not self.students_cache or features is None:
            return None, 0.0
            
        best_match = None
        best_score = -1
        
        for student_id, student_data in self.students_cache.items():
            student_name = student_data['name']
            stored_features = student_data['features']
            
            # Calculate similarity with each stored feature set
            similarities = []
            for stored_feature in stored_features:
                try:
                    # Multiple similarity metrics
                    cosine_sim = self.cosine_similarity(features, stored_feature)
                    euclidean_sim = self.euclidean_distance(features, stored_feature)
                    correlation_sim = self.correlation_similarity(features, stored_feature)
                    
                    # Weighted combination of similarities
                    combined_score = (
                        cosine_sim * 0.4 +           # Cosine similarity (40%)
                        euclidean_sim * 0.35 +       # Euclidean similarity (35%)
                        correlation_sim * 0.25       # Correlation similarity (25%)
                    )
                    
                    similarities.append(combined_score)
                    
                except Exception as e:
                    continue
            
            if similarities:
                # Use top 3 similarities for robustness
                similarities = sorted(similarities, reverse=True)
                top_similarities = similarities[:min(3, len(similarities))]
                avg_similarity = np.mean(top_similarities)
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_match = (student_id, student_name)
                    
        if best_score >= threshold:
            return best_match, best_score
        else:
            return None, best_score
            
    def live_recognition(self, class_id, threshold=0.7):
        """Live face recognition with comprehensive improvements"""
        print(f"\n🎯 Starting live recognition for {class_id}")
        
        # Load students
        self.load_class_students(class_id)
        if not self.students_cache:
            print("❌ No students found in class")
            return
            
        print(f"🔍 Threshold: {threshold} | Press 'q' to quit, 'r' to reload students")
        print(f"📈 Improved recognition with normalization and quality control")
        print(f"⏱️  Recognition results printed every 1 second")
        
        cap = cv2.VideoCapture(0)
        
        # Set camera to 30 FPS if possible
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Camera FPS: {actual_fps}")
        
        # Variables for continuous recognition
        current_recognition = None
        current_confidence = 0.0
        current_quality = 0.0
        last_print_time = 0
        print_interval = 1.0  # Print every 1 second
        
        # Statistics
        frame_count = 0
        recognition_count = 0
        successful_recognition = 0
        quality_sum = 0.0
        
        # Stability tracking
        recognition_history = []
        history_length = 5  # Track last 5 recognitions for stability
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time()
            frame_count += 1
            
            # Draw alignment guide
            frame_with_guide = self.draw_live_guide(frame)
            
            # Extract features and recognize face every frame
            features, quality = self.extract_landmarks(frame)
            
            if features is not None and quality:
                match, confidence = self.recognize_face(features, threshold)
                recognition_count += 1
                quality_sum += quality
                
                if match:
                    student_id, student_name = match
                    current_recognition = student_name
                    current_confidence = confidence
                    successful_recognition += 1
                    
                    # Add to history for stability
                    recognition_history.append(student_name)
                else:
                    current_recognition = "Unknown"
                    current_confidence = confidence
                    recognition_history.append("Unknown")
                    
                current_quality = quality if isinstance(quality, (int, float)) else 0.0
            else:
                current_recognition = "No face detected" if not features else "Poor quality"
                current_confidence = 0.0
                current_quality = quality if isinstance(quality, (int, float)) else 0.0
                recognition_history.append("No face")
            
            # Keep history to reasonable length
            if len(recognition_history) > history_length:
                recognition_history.pop(0)
            
            # Print recognition result every second
            if current_time - last_print_time >= print_interval:
                if current_recognition:
                    # Calculate stability (how consistent recent recognitions are)
                    if len(recognition_history) >= 3:
                        stability = self.calculate_stability(recognition_history[-3:])
                        stability_text = f" [Stability: {stability:.1f}%]"
                    else:
                        stability_text = ""
                    
                    if current_recognition not in ["No face detected", "Unknown", "Poor quality"]:
                        print(f"🔍 RECOGNIZED: {current_recognition} (conf: {current_confidence:.3f}, qual: {current_quality:.3f}){stability_text}")
                    elif current_recognition == "Unknown":
                        print(f"❓ Unknown face (conf: {current_confidence:.3f}, qual: {current_quality:.3f}){stability_text}")
                    else:
                        print(f"👁️  {current_recognition} (qual: {current_quality:.3f})")
                
                last_print_time = current_time
            
            # Draw landmarks on frame if available
            if features is not None:
                # Convert back to landmark format for visualization (approximate)
                self._draw_quality_indicators(frame_with_guide, current_quality, current_confidence)
            
            # Display current recognition on frame
            self._display_improved_overlay(
                frame_with_guide, current_recognition, current_confidence, 
                current_quality, threshold, frame_count, recognition_count, 
                successful_recognition, quality_sum, start_time, current_time,
                recognition_history
            )
            
            cv2.imshow('Live Recognition - Enhanced', frame_with_guide)
            
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
        avg_quality = quality_sum / recognition_count if recognition_count > 0 else 0
        
        print(f"\n📊 Enhanced Session Summary:")
        print(f"   Duration: {total_time:.1f}s")
        print(f"   Total frames: {frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Recognition attempts: {recognition_count}")
        print(f"   Successful recognitions: {successful_recognition}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Average quality: {avg_quality:.3f}")
        
    def draw_live_guide(self, frame):
        """Draw minimal alignment guide for live recognition"""
        h, w, _ = frame.shape
        frame_copy = frame.copy()
        
        # Draw minimal center cross
        center_x, center_y = w // 2, h // 2 - 20
        cross_size = 10
        cv2.line(frame_copy, (center_x-cross_size, center_y), (center_x+cross_size, center_y), (255, 255, 255), 1)
        cv2.line(frame_copy, (center_x, center_y-cross_size), (center_x, center_y+cross_size), (255, 255, 255), 1)
        
        return frame_copy
        
    def calculate_stability(self, recent_recognitions):
        """Calculate stability percentage of recent recognitions"""
        if not recent_recognitions:
            return 0.0
            
        # Count most common recognition
        from collections import Counter
        counts = Counter(recent_recognitions)
        most_common_count = counts.most_common(1)[0][1]
        
        return (most_common_count / len(recent_recognitions)) * 100
        
    def _draw_quality_indicators(self, frame, quality, confidence):
        """Draw quality and confidence indicators"""
        h, w, _ = frame.shape
        
        # Quality bar (left side)
        bar_width, bar_height = 20, 200
        bar_x, bar_y = 20, h - bar_height - 50
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Quality fill
        fill_height = int(bar_height * quality)
        quality_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height), 
                     (bar_x + bar_width, bar_y + bar_height), quality_color, -1)
        
        # Labels
        cv2.putText(frame, "Q", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Confidence bar (right side)
        bar_x2 = w - 40
        cv2.rectangle(frame, (bar_x2, bar_y), (bar_x2 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        fill_height2 = int(bar_height * confidence)
        conf_color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255) if confidence > 0.6 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x2, bar_y + bar_height - fill_height2), 
                     (bar_x2 + bar_width, bar_y + bar_height), conf_color, -1)
        
        cv2.putText(frame, "C", (bar_x2, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        
    def _display_improved_overlay(self, frame, current_recognition, current_confidence, 
                                current_quality, threshold, frame_count, recognition_count, 
                                successful_recognition, quality_sum, start_time, current_time,
                                recognition_history):
        """Enhanced overlay display with comprehensive information"""
        # Create semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Show current recognition with better color coding
        if current_recognition:
            if current_recognition not in ["No face detected", "Unknown", "Poor quality"]:
                color = (0, 255, 0)  # Green for recognized
                text = f"✅ RECOGNIZED: {current_recognition}"
            elif current_recognition == "Unknown":
                color = (0, 165, 255)  # Orange for unknown
                text = f"❓ Unknown Face"
            else:
                color = (128, 128, 128)  # Gray for no face/poor quality
                text = f"👁️ {current_recognition}"
                
            cv2.putText(frame, text, (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show confidence and quality
            if current_confidence > 0:
                cv2.putText(frame, f"Confidence: {current_confidence:.3f}", 
                           (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if current_quality > 0:
                cv2.putText(frame, f"Quality: {current_quality:.3f}", 
                           (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show stability indicator
        if len(recognition_history) >= 3:
            stability = self.calculate_stability(recognition_history[-3:])
            stability_color = (0, 255, 0) if stability > 80 else (0, 165, 255) if stability > 50 else (0, 0, 255)
            cv2.putText(frame, f"Stability: {stability:.0f}%", 
                       (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1)
        
        # Show statistics on the right side
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if recognition_count > 0:
            success_rate = (successful_recognition / recognition_count) * 100
            avg_quality = quality_sum / recognition_count
            cv2.putText(frame, f"Success: {success_rate:.1f}%", 
                       (frame.shape[1] - 120, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Avg Qual: {avg_quality:.2f}", 
                       (frame.shape[1] - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show threshold
        cv2.putText(frame, f"Thresh: {threshold:.2f}", 
                   (frame.shape[1] - 120, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions at bottom
        cv2.putText(frame, "Enhanced Recognition | Q: quit | R: reload | White cross: face center", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Real-time Face Recognition System')
    parser.add_argument('--mode', choices=['register', 'live'], required=True,
                       help='Mode: register new student or live recognition')
    parser.add_argument('--class-id', required=True,
                       help='Class identifier')
    parser.add_argument('--student-id', help='Student ID (for registration mode)')
    parser.add_argument('--student-name', help='Student name (for registration mode)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Recognition threshold (0.0-1.0, recommended: 0.6-0.8)')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("❌ Threshold must be between 0.0 and 1.0")
        return
    
    # Provide threshold recommendations
    if args.threshold > 0.9:
        print(f"⚠️  High threshold ({args.threshold}) - may cause false negatives")
    elif args.threshold < 0.5:
        print(f"⚠️  Low threshold ({args.threshold}) - may cause false positives")
    else:
        print(f"✅ Good threshold ({args.threshold}) for balanced recognition")
    
    system = SimpleFaceRecognizer()
    
    if args.mode == 'register':
        if not args.student_id or not args.student_name:
            print("❌ For registration mode, need --student-id and --student-name")
            print("Example: python facerec.py --mode register --class-id CS101 --student-id S001 --student-name 'John Doe'")
            return
            
        print("\n🎯 ENHANCED REGISTRATION MODE")
        print("   • Face alignment guide with quality control")
        print("   • Normalized landmarks for better accuracy")
        print("   • Multiple similarity metrics")
        print("   • Quality-based capture filtering")
        
        system.register_student(args.class_id, args.student_id, args.student_name)
        
    elif args.mode == 'live':
        print("\n🎯 ENHANCED LIVE RECOGNITION MODE")
        print("   • Real-time quality assessment")
        print("   • Stability tracking")
        print("   • Multiple similarity metrics")
        print("   • Visual quality indicators")
        
        system.live_recognition(args.class_id, args.threshold)

if __name__ == "__main__":
    main()
