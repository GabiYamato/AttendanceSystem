# Face Recognition Attendance System

A comprehensive attendance system using MediaPipe face landmarks for robust face recognition, built with machine learning embeddings and Firebase backend.

## 🌟 Features

- **Advanced Face Recognition**: Uses MediaPipe landmarks + deep learning embeddings
- **Robust Training Pipeline**: Train on celebrity datasets with triplet/contrastive loss
- **Real-time Attendance**: Live face recognition with automatic attendance marking
- **Web Dashboard**: Modern Flask-based interface for management
- **Firebase Integration**: Cloud database for students and attendance records
- **Data Augmentation**: Improves model robustness with landmark augmentation
- **Threshold Tuning**: Configurable similarity thresholds for recognition accuracy

## 🏗️ System Architecture

# Simple Face Recognition Attendance System

A streamlined face recognition attendance system using MediaPipe face landmarks and PyTorch embeddings.

## 🌟 Features

- **MediaPipe Integration**: Uses 478 face landmarks for accurate face detection
- **Deep Learning**: PyTorch-based embedding model trained on celebrity faces
- **Firebase Backend**: Real-time database for student management and attendance tracking
- **Class-based Organization**: Supports multiple classes and students
- **Live Recognition**: Real-time face recognition with attendance marking
- **Simplified Architecture**: Just 4 main files for the entire system

## 🏗️ Project Structure

```
AttendanceSystem/
├── download_dataset.py           # Download and process celebrity dataset
├── train_simple.py              # Train face embedding model
├── firebase_simple.py           # Firebase operations
├── attendance_simple.py         # Main registration and live recognition
├── mediapipeface.py            # Your custom MediaPipe implementation
├── data/                       # Training dataset
├── trained_models/             # Saved model weights
└── requirements.txt            # Python dependencies
```

## 📋 Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- PyTorch
- Firebase Admin SDK
- Flask
- NumPy, Pandas, Scikit-learn

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
git clone <your-repo>
cd AttendanceSystem

# Create virtual environment
python3 -m venv macenv
source macenv/bin/activate  # On Windows: macenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Firestore Database
3. Generate service account credentials (JSON file)
4. Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your Firebase credentials
```

### 3. Download MediaPipe Model

Download the face landmark model:
```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -O face_landmarker_v2_with_blendshapes.task
```

### 4. Initialize System

```bash
# Setup Firebase collections and directories
python main.py --mode setup
```

## 📖 Usage Guide

### Step 1: Train the Embedding Model

```bash
# Train MLP model with triplet loss
python main.py --mode train --model mlp --epochs 100

# Train Transformer model
python main.py --mode train --model transformer --epochs 100
```

### Step 2: Register Students

```bash
# Interactive registration
python main.py --mode register

# Command line registration
python main.py --mode register --student_id "STU001" --student_name "John Doe" --email "john@example.com"
```

### Step 3: Live Recognition

```bash
# Start live attendance marking
python main.py --mode live --session_id "morning_class"

# Use specific model
python main.py --mode live --model transformer
```

### Step 4: Web Dashboard

```bash
# Start web interface
python app.py
# Open http://localhost:5000
```

## 🎯 Training Pipeline

### Celebrity Dataset Preparation

1. **Download Dataset**: Use VGGFace2, MS-Celeb-1M, or LFW
2. **Extract Landmarks**: Run MediaPipe on all images
3. **Preprocessing**: Normalize landmarks (center + scale)
4. **Augmentation**: Rotation, flip, noise injection

### Model Training

```python
# Example training script
from src.models.train_model import train_model

# Train with triplet loss
model = train_model(
    model_type="mlp",      # or "transformer"
    loss_type="triplet",   # or "contrastive"
    num_epochs=100
)
```

### Training Data Format

```python
# Landmarks shape: (N, 1404) - 468 landmarks × 3 coordinates
# Labels shape: (N,) - Person IDs
landmarks, labels = load_celebrity_dataset()
```

## 🔧 Configuration

Edit `config/config.py` for system settings:

```python
# Recognition thresholds
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity for recognition
MIN_REGISTRATION_FRAMES = 5  # Minimum frames per student
MAX_REGISTRATION_FRAMES = 10 # Maximum frames per student

# Model settings
EMBEDDING_DIMENSION = 128    # Size of face embeddings
BATCH_SIZE = 32             # Training batch size
LEARNING_RATE = 0.001       # Learning rate for training
```

## 📊 Face Recognition Process

### 1. Landmark Extraction
```python
processor = LandmarkProcessor()
landmarks = processor.extract_landmarks_from_image(image)
normalized = processor.normalize_landmarks(landmarks)
```

### 2. Embedding Generation
```python
model = FaceEmbeddingMLP()
embedding = model(torch.FloatTensor(landmarks))
```

### 3. Similarity Matching
```python
similarity = cosine_similarity_score(query_embedding, stored_embedding)
if similarity >= SIMILARITY_THRESHOLD:
    return student_id
```

### 4. Attendance Marking
```python
if recognized_student:
    firebase_manager.mark_attendance(student_id, session_id)
```

## 🌐 Web Dashboard Features

- **Dashboard**: Real-time attendance statistics
- **Students**: Manage student profiles and registrations
- **Attendance**: View and export attendance records
- **Registration**: Web-based student registration with face capture

### Dashboard Screenshots

- Live attendance monitoring
- Student management interface
- Attendance history and reports
- Face registration workflow

## 📁 Project Structure

```
AttendanceSystem/
├── main.py                 # Main CLI application
├── app.py                  # Flask web dashboard
├── mediapipeface.py        # Original MediaPipe demo
├── config/
│   └── config.py          # System configuration
├── src/
│   ├── models/
│   │   ├── embedding_model.py    # Neural network models
│   │   └── train_model.py        # Training pipeline
│   ├── utils/
│   │   ├── landmark_processor.py # MediaPipe processing
│   │   └── face_recognition_engine.py # Main recognition engine
│   └── database/
│       └── firebase_manager.py   # Firebase operations
├── templates/             # HTML templates for web interface
├── data/                 # Dataset storage
│   ├── celebrity_dataset/
│   └── student_images/
├── trained_models/       # Saved model weights
└── requirements.txt      # Python dependencies
```

## 🔬 Technical Details

### Landmark Preprocessing

1. **Centering**: Use nose tip as origin point
2. **Scaling**: Normalize by eye corner distance  
3. **Flattening**: Convert (468,3) to (1404,) vector

### Model Architecture

**MLP Model:**
```
Input (1404) → Dense(512) → BatchNorm → ReLU → Dropout
→ Dense(256) → BatchNorm → ReLU → Dropout  
→ Dense(128) → L2 Normalize → Output
```

**Transformer Model:**
```
Input (468,3) → Linear Projection → Positional Encoding
→ Transformer Encoder Layers → Global Average Pooling
→ Dense → L2 Normalize → Output (128)
```

### Loss Functions

**Triplet Loss:**
```python
loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
```

**Contrastive Loss:**
```python
loss = (1-Y) * D² + Y * max(0, margin - D)²
```

## 🎛️ Command Line Interface

```bash
# Setup system
python main.py --mode setup

# Train model
python main.py --mode train --model mlp --epochs 100 --loss triplet

# Register student
python main.py --mode register --student_id "STU001" --student_name "John Doe"

# Live recognition
python main.py --mode live --session_id "morning_class" --model mlp

# Interactive mode (no arguments)
python main.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera index in config
2. **No face detected**: Ensure good lighting and face visibility
3. **Low recognition accuracy**: Retrain model or adjust threshold
4. **Firebase errors**: Verify credentials and project configuration

### Performance Optimization

1. **GPU Training**: Install CUDA-enabled PyTorch for faster training
2. **Batch Processing**: Increase batch size if memory allows
3. **Model Optimization**: Use TensorRT or ONNX for inference speed
4. **Caching**: Enable embedding caching for repeated recognitions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's ML framework for face landmark detection
- **PyTorch**: Deep learning framework for embedding models
- **Firebase**: Backend database and authentication
- **Flask**: Web framework for dashboard interface
- **Bootstrap**: Frontend UI components

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration guide

---

**Built with ❤️ for educational and research purposes**
