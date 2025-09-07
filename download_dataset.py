#!/usr/bin/env python3
"""
Simple script to download and prepare celebrity dataset for face recognition training
"""
import os
import numpy as np
from datasets import load_dataset
import mediapipe as mp
from pathlib import Path
import pickle
import cv2
from PIL import Image

def setup_mediapipe():
    """Initialize MediaPipe face detection and landmarks"""
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5
    )
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return face_detection, face_mesh

def extract_landmarks(image, face_detection, face_mesh):
    """Extract 478 face landmarks from image"""
    try:
        # Convert PIL to opencv format
        if isinstance(image, Image.Image):
            image_rgb = np.array(image)
        else:
            image_rgb = image
            
        # Detect face first
        detection_results = face_detection.process(image_rgb)
        if not detection_results.detections:
            return None
            
        # Extract landmarks
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
            
        # Get the first face landmarks (478 points)
        landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array [478, 3] (x, y, z)
        landmark_points = []
        for landmark in landmarks.landmark:
            landmark_points.extend([landmark.x, landmark.y, landmark.z])
            
        return np.array(landmark_points)  # Shape: (1434,) = 478 * 3
        
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return None

def download_and_process_dataset():
    """Download celebrity dataset and extract landmarks"""
    print("🚀 Starting celebrity dataset download and processing...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Setup MediaPipe
    print("🔧 Setting up MediaPipe...")
    face_detection, face_mesh = setup_mediapipe()
    
    try:
        # Load dataset
        print("📥 Loading celebrity dataset...")
        dataset = load_dataset("theneuralmaze/celebrity_faces", split="train", streaming=True)
        
        landmarks_list = []
        labels_list = []
        processed = 0
        max_samples = 1000  # Limit for faster processing
        max_per_person = 5
        person_counts = {}
        
        print(f"🎯 Processing up to {max_samples} images...")
        
        for i, sample in enumerate(dataset):
            if processed >= max_samples:
                break
                
            try:
                # Get label and image
                label = sample.get('label', 'unknown')
                image = sample.get('image')
                
                if image is None:
                    continue
                    
                # Limit samples per person
                if person_counts.get(label, 0) >= max_per_person:
                    continue
                    
                # Extract landmarks
                landmarks = extract_landmarks(image, face_detection, face_mesh)
                if landmarks is not None:
                    landmarks_list.append(landmarks)
                    labels_list.append(label)
                    person_counts[label] = person_counts.get(label, 0) + 1
                    processed += 1
                    
                    if processed % 50 == 0:
                        print(f"✅ Processed {processed} images from {len(person_counts)} people")
                        
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                continue
        
        if not landmarks_list:
            raise Exception("No landmarks extracted from dataset")
            
        # Convert to numpy arrays
        landmarks_array = np.array(landmarks_list)
        labels_array = np.array(labels_list)
        
        print(f"🎉 Successfully processed {len(landmarks_list)} samples from {len(set(labels_list))} people")
        print(f"📊 Landmarks shape: {landmarks_array.shape}")
        
        # Save processed dataset
        dataset_path = data_dir / "celebrity_landmarks.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump({
                'landmarks': landmarks_array,
                'labels': labels_array,
                'num_samples': len(landmarks_list),
                'num_people': len(set(labels_list))
            }, f)
            
        print(f"💾 Saved processed dataset to: {dataset_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading/processing dataset: {e}")
        return False

if __name__ == "__main__":
    success = download_and_process_dataset()
    if success:
        print("✅ Dataset ready for training!")
    else:
        print("❌ Dataset preparation failed!")
