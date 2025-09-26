#!/usr/bin/env python3
"""
Quick test script for QR code generation
"""
import qrcode
from PIL import Image

def test_qr_generation():
    """Test QR code generation"""
    print("🧪 Testing QR Code Generation")
    
    # Test class IDs
    test_classes = ["cs104", "math101", "physics201", "hackathon_class"]
    
    for class_id in test_classes:
        print(f"📱 Generating QR for: {class_id}")
        
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(f"CLASS:{class_id}")
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save to file
        filename = f"qr_{class_id}.png"
        img.save(filename)
        print(f"✅ Saved: {filename}")
    
    print("\n🎯 QR Code Test Complete!")
    print("📋 Next steps:")
    print("1. Start the attendance system: streamlit run finalserver.py")
    print("2. Use generated QR codes to test scanning")
    print("3. Make sure students are registered first using face_recognizer.py")

if __name__ == "__main__":
    test_qr_generation()
