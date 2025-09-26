# Smart Attendance System - Usage Instructions

## Overview
This is a step-by-step attendance system that combines QR code verification and face recognition to mark student attendance automatically.

## Features
- ✅ QR Code class verification
- 👤 Face recognition for student identification  
- 📊 Real-time attendance marking
- 🔒 Firebase integration for data storage
- 📱 Streamlit web interface

## Prerequisites
1. Students must be registered using `face_recognizer.py` first
2. Firebase service account credentials (`firebase-service-account.json`)
3. Camera access for QR scanning and face recognition

## How to Use

### Step 1: Start the System
```bash
# Activate virtual environment
source macenv/bin/activate

# Run the Streamlit application
streamlit run finalserver.py
```

### Step 2: QR Code Verification
1. **Generate QR Code** (using separate script): 
   ```bash
   python3 generate_qr.py cs104
   ```
   - This creates a QR code file (e.g., `qr_cs104.png`)
   - Print or display the QR code for students to scan

2. **Scan QR Code or Manual Entry**:
   - Students click "Start QR Camera Scan" and point camera at QR code
   - **Alternative**: Enter class ID manually (e.g., "cs104")
   - System verifies/creates class in Firebase
   - Classes are auto-created if they don't exist

### Step 3: Face Recognition & Attendance
1. **Automatic Student Detection**:
   - Camera activates for face recognition
   - System loads all registered students for the class
   - Real-time face detection and matching

2. **Attendance Marking**:
   - Requires 5 consecutive detections for accuracy
   - Prevents duplicate attendance (one per day)
   - Shows success message and balloons animation
   - Stores attendance with timestamp in Firebase

### Step 4: Controls
- **Recognition Tolerance**: Adjust slider (0.3-0.8)
  - Lower = Stricter matching
  - Higher = More lenient matching
- **Stop Recognition**: Click stop button anytime
- **Back Navigation**: Return to QR scan step

## System Flow
```
1. QR Code Scan → 2. Class Verification → 3. Face Recognition → 4. Attendance Marked
```

## Troubleshooting

### Common Issues:
1. **"No students registered"**: Use `face_recognizer.py` to register students first
2. **"Error verifying class code"**: Classes are auto-created, check Firebase connection
3. **Camera not working**: Check camera permissions
4. **QR code not detected**: Ensure good lighting and QR code visibility
5. **Face not recognized**: Adjust tolerance or re-register student

### Generate QR Codes:
```bash
python3 generate_qr.py cs104
python3 generate_qr.py math101 --output math_class_qr.png
```

### Registration Command:
```bash
python3 face_recognizer.py --mode register --class_id cs104 --student_id 123 --name "John Doe"
```

### Detection Testing:
```bash
python3 face_recognizer.py --mode detect --class_id cs104
```

## Database Structure
```
Firebase Firestore:
/classes/{class_id}/
  ├── students/{student_id}
  │   ├── name, class_id, registered_at
  │   └── encodings/{frame_id}
  │       └── encoding[], frame_number
  └── attendance/{date}/
      └── students/{student_id}
          └── marked_at, status, student_name
```

## Security Features
- Class verification before attendance
- Multiple face encodings for accuracy
- Consecutive detection requirement
- Duplicate attendance prevention
- Firebase security rules

## Performance Tips
- Ensure good lighting for face recognition
- Position camera at eye level
- Process every 3rd frame for performance
- Use tolerance 0.4-0.6 for best results
- Limit to 30 students per class for optimal speed

## Requirements
All dependencies in `requirements.txt`:
- streamlit
- opencv-python
- face-recognition
- qrcode[pil]
- pyzbar
- firebase-admin
- numpy
- pillow

## Support
- Check Firebase console for data verification
- Monitor Streamlit logs for debugging
- Ensure all students are properly registered
- Test QR codes before class sessions
