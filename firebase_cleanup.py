"""
Firebase Cleanup Script
Removes all unnecessary collections and reinitializes clean schema
"""
import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

class FirebaseCleanup:
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

    def list_all_collections(self):
        """List all root-level collections"""
        try:
            collections = self.db.collections()
            collection_names = [col.id for col in collections]
            print(f"📋 Found {len(collection_names)} root collections:")
            for name in collection_names:
                print(f"   - {name}")
            return collection_names
        except Exception as e:
            print(f"❌ Error listing collections: {e}")
            return []

    def count_documents_in_collection(self, collection_name):
        """Count documents in a collection"""
        try:
            docs = list(self.db.collection(collection_name).stream())
            return len(docs)
        except Exception as e:
            print(f"❌ Error counting documents in {collection_name}: {e}")
            return 0

    def delete_collection(self, collection_name, batch_size=100):
        """Delete all documents in a collection"""
        try:
            collection_ref = self.db.collection(collection_name)
            docs = collection_ref.limit(batch_size).stream()
            deleted = 0
            
            for doc in docs:
                # Delete subcollections first
                subcollections = doc.reference.collections()
                for subcol in subcollections:
                    self.delete_collection(f"{collection_name}/{doc.id}/{subcol.id}")
                
                # Delete the document
                doc.reference.delete()
                deleted += 1
                
            if deleted > 0:
                print(f"🗑️ Deleted {deleted} documents from {collection_name}")
                # Recursively delete remaining documents
                self.delete_collection(collection_name, batch_size)
            else:
                print(f"✅ Collection {collection_name} is now empty")
                
        except Exception as e:
            print(f"❌ Error deleting collection {collection_name}: {e}")

    def show_current_schema(self):
        """Show current Firebase schema"""
        print("\n📊 Current Firebase Schema:")
        print("=" * 50)
        
        collections = self.list_all_collections()
        
        for collection_name in collections:
            doc_count = self.count_documents_in_collection(collection_name)
            print(f"\n📁 Collection: {collection_name} ({doc_count} documents)")
            
            # Sample a few documents to show structure
            try:
                docs = list(self.db.collection(collection_name).limit(3).stream())
                for i, doc in enumerate(docs):
                    data = doc.to_dict()
                    print(f"   📄 Document {i+1}: {doc.id}")
                    
                    # Show field types
                    for field, value in list(data.items())[:5]:  # Show first 5 fields
                        field_type = type(value).__name__
                        if field_type == 'list':
                            field_type = f"list[{len(value)}]"
                        elif field_type == 'dict':
                            field_type = f"dict[{len(value)}]"
                        print(f"      - {field}: {field_type}")
                    
                    if len(data) > 5:
                        print(f"      ... and {len(data) - 5} more fields")
                        
                    # Check for subcollections
                    subcollections = doc.reference.collections()
                    for subcol in subcollections:
                        subcol_docs = list(subcol.limit(1).stream())
                        print(f"      📁 Subcollection: {subcol.id} ({len(list(subcol.stream()))} docs)")
                        
            except Exception as e:
                print(f"   ❌ Error reading documents: {e}")

    def cleanup_all(self):
        """Remove all collections"""
        print("\n🧹 Starting Firebase cleanup...")
        
        collections = self.list_all_collections()
        
        if not collections:
            print("✅ No collections found - database is already clean")
            return
            
        print(f"\n⚠️ This will delete ALL {len(collections)} collections:")
        for name in collections:
            doc_count = self.count_documents_in_collection(name)
            print(f"   - {name} ({doc_count} documents)")
            
        confirm = input("\n❓ Are you sure you want to delete ALL data? (type 'DELETE' to confirm): ")
        
        if confirm == 'DELETE':
            print("\n🗑️ Deleting all collections...")
            for collection_name in collections:
                print(f"\n🗑️ Deleting collection: {collection_name}")
                self.delete_collection(collection_name)
            print("\n✅ All collections deleted!")
        else:
            print("❌ Cleanup cancelled")

    def initialize_clean_schema(self):
        """Initialize clean schema for simplified system"""
        print("\n🏗️ Initializing clean schema...")
        
        # Create a sample class with proper structure
        sample_class_id = "SAMPLE_CLASS_2024"
        sample_student_id = "SAMPLE_STUDENT_001"
        
        try:
            # Create class document
            class_ref = self.db.collection('classes').document(sample_class_id)
            class_ref.set({
                'name': 'Sample Computer Science Class',
                'description': 'Sample class for testing attendance system',
                'created_at': datetime.now(),
                'is_active': True,
                'instructor': 'Sample Instructor'
            })
            
            # Create student document
            student_ref = class_ref.collection('students').document(sample_student_id)
            student_ref.set({
                'name': 'Sample Student',
                'class_id': sample_class_id,
                'registered_at': datetime.now(),
                'is_active': True
            })
            
            # Create sample embedding
            embedding_ref = student_ref.collection('embeddings').document('frame_0')
            embedding_ref.set({
                'embedding': [0.1] * 128,  # Sample 128-dim embedding
                'frame_number': 0,
                'created_at': datetime.now()
            })
            
            # Create sample attendance
            today = datetime.now().strftime('%Y-%m-%d')
            attendance_ref = student_ref.collection('attendance').document(today)
            attendance_ref.set({
                'date': today,
                'timestamp': datetime.now(),
                'confidence': 0.95,
                'status': 'present'
            })
            
            print("✅ Clean schema initialized!")
            print(f"📁 Created sample structure:")
            print(f"   classes/{sample_class_id}/")
            print(f"   ├── students/{sample_student_id}/")
            print(f"   │   ├── embeddings/frame_0")
            print(f"   │   └── attendance/{today}")
            print(f"\n💡 You can delete this sample data once you register real students")
            
        except Exception as e:
            print(f"❌ Error initializing schema: {e}")

def main():
    cleanup = FirebaseCleanup()
    
    while True:
        print("\n" + "="*60)
        print("🔥 Firebase Cleanup & Schema Manager")
        print("="*60)
        print("1. Show current schema")
        print("2. Cleanup all collections")
        print("3. Initialize clean schema")
        print("4. Full reset (cleanup + initialize)")
        print("5. Exit")
        print("="*60)
        
        choice = input("👉 Select option (1-5): ").strip()
        
        if choice == '1':
            cleanup.show_current_schema()
        elif choice == '2':
            cleanup.cleanup_all()
        elif choice == '3':
            cleanup.initialize_clean_schema()
        elif choice == '4':
            cleanup.cleanup_all()
            cleanup.initialize_clean_schema()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()
