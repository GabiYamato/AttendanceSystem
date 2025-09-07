"""
Simple Embedding Model Training
Single file for training face embedding model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
import argparse

class SimpleEmbeddingModel(nn.Module):
    """Simple MLP for face embedding"""
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
            nn.LayerNorm(embedding_dim)  # Keep normalization for stability
        )
        
    def forward(self, x):
        return F.normalize(self.layers(x), p=2, dim=1)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet loss function"""
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def create_triplets(embeddings, labels, num_triplets=1000):
    """Create triplet pairs for training"""
    triplets = []
    unique_labels = list(set(labels))
    
    for _ in range(num_triplets):
        # Random anchor
        anchor_label = np.random.choice(unique_labels)
        anchor_indices = [i for i, l in enumerate(labels) if l == anchor_label]
        if len(anchor_indices) < 2:
            continue
            
        anchor_idx = np.random.choice(anchor_indices)
        
        # Positive (same person)
        positive_indices = [i for i in anchor_indices if i != anchor_idx]
        if not positive_indices:
            continue
        positive_idx = np.random.choice(positive_indices)
        
        # Negative (different person)
        negative_labels = [l for l in unique_labels if l != anchor_label]
        if not negative_labels:
            continue
        negative_label = np.random.choice(negative_labels)
        negative_indices = [i for i, l in enumerate(labels) if l == negative_label]
        negative_idx = np.random.choice(negative_indices)
        
        triplets.append((anchor_idx, positive_idx, negative_idx))
    
    return triplets

def load_dataset():
    """Load preprocessed celebrity dataset"""
    dataset_path = Path("data/celebrity_landmarks.pkl")
    
    if not dataset_path.exists():
        print("❌ Dataset not found! Run download_dataset.py first")
        return None, None
        
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
            
        landmarks = data['landmarks']
        labels = data['labels']
        
        print(f"✅ Loaded dataset: {landmarks.shape[0]} samples from {len(set(labels))} people")
        return landmarks, labels
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None

def train_model(epochs=10, learning_rate=0.001):
    """Train the embedding model"""
    print("🚀 Starting model training...")
    
    # Load dataset
    landmarks, labels = load_dataset()
    if landmarks is None:
        return False
        
    # Convert to tensors
    landmarks_tensor = torch.FloatTensor(landmarks)
    
    # Initialize model
    model = SimpleEmbeddingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"📊 Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Create triplets for this epoch
        triplets = create_triplets(landmarks, labels, num_triplets=min(1000, len(landmarks)//3))
        
        for i, (anchor_idx, pos_idx, neg_idx) in enumerate(triplets):
            optimizer.zero_grad()
            
            # Get embeddings
            anchor_emb = model(landmarks_tensor[anchor_idx:anchor_idx+1])
            pos_emb = model(landmarks_tensor[pos_idx:pos_idx+1])  
            neg_emb = model(landmarks_tensor[neg_idx:neg_idx+1])
            
            # Compute loss
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if epoch % 2 == 0:  # Print every 2 epochs to reduce output
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")    # Save model
    model_dir = Path("trained_models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "simple_face_model.pth"
    
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    
    return True

def test_model():
    """Test the trained model"""
    landmarks, labels = load_dataset()
    if landmarks is None:
        return
        
    model = SimpleEmbeddingModel()
    model_path = Path("trained_models/simple_face_model.pth")
    
    if not model_path.exists():
        print("❌ No trained model found!")
        return
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Test on a few samples
    with torch.no_grad():
        sample_landmarks = torch.FloatTensor(landmarks[:5])
        embeddings = model(sample_landmarks)
        
        print(f"✅ Model test successful!")
        print(f"📊 Input shape: {sample_landmarks.shape}")
        print(f"📊 Output shape: {embeddings.shape}")
        print(f"📊 Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print(f"📊 Embedding norm: {embeddings.norm(dim=1).mean():.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        success = train_model(epochs=args.epochs, learning_rate=args.lr)
        if success:
            test_model()
    else:
        test_model()
