#!/usr/bin/env python3
"""
Simple QR Code Generator for Class Codes
Creates QR codes that can be scanned by the attendance system
"""
import qrcode
import argparse

def generate_class_qr(class_id, output_file=None):
    """Generate QR code for a class"""
    print(f"📱 Generating QR code for class: {class_id}")
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(f"CLASS:{class_id}")
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    if output_file is None:
        output_file = f"qr_{class_id}.png"
    
    img.save(output_file)
    print(f"✅ QR code saved as: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate QR codes for class attendance")
    parser.add_argument('class_id', help="Class ID to generate QR code for")
    parser.add_argument('--output', '-o', help="Output file name (default: qr_{class_id}.png)")
    
    args = parser.parse_args()
    
    try:
        output_file = generate_class_qr(args.class_id, args.output)
        print(f"\n🎯 Success! QR code generated for class '{args.class_id}'")
        print(f"📄 File: {output_file}")
        print("\n📋 Next steps:")
        print("1. Print or display the QR code")
        print("2. Students scan this QR code in the attendance system")
        print("3. Make sure students are registered first using: python3 face_recognizer.py --mode register")
        
    except Exception as e:
        print(f"❌ Error generating QR code: {e}")

if __name__ == "__main__":
    main()
