#!/usr/bin/env python3
"""
Student Data Management Script
Simple script to list, delete, or clear student data
"""
import argparse
from firebase_simple import SimpleFirebase

def list_students(firebase, class_id):
    """List all students in a class"""
    print(f"\n📋 Students in class {class_id}:")
    print("-" * 50)
    
    students = firebase.list_class_students(class_id)
    if not students:
        print("   No students found")
        return
    
    for i, student in enumerate(students, 1):
        status = "✅ Active" if student['is_active'] else "❌ Inactive"
        reg_date = student['registered_at'].strftime('%Y-%m-%d %H:%M') if student['registered_at'] else "Unknown"
        print(f"   {i}. {student['name']} (ID: {student['student_id']})")
        print(f"      Status: {status} | Registered: {reg_date}")
        print()

def delete_student(firebase, class_id, student_id):
    """Delete a specific student"""
    print(f"\n🗑️  Deleting student {student_id} from class {class_id}...")
    
    # First check if student exists
    students = firebase.list_class_students(class_id)
    student_found = False
    student_name = ""
    
    for student in students:
        if student['student_id'] == student_id:
            student_found = True
            student_name = student['name']
            break
    
    if not student_found:
        print(f"❌ Student {student_id} not found in class {class_id}")
        return False
    
    # Confirm deletion
    confirm = input(f"⚠️  Are you sure you want to delete {student_name} ({student_id})? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Deletion cancelled")
        return False
    
    success = firebase.delete_student(class_id, student_id)
    if success:
        print(f"✅ Successfully deleted {student_name}")
    return success

def clear_class(firebase, class_id):
    """Clear all students from a class"""
    print(f"\n🧹 Clearing all student data from class {class_id}...")
    
    # Show current students
    students = firebase.list_class_students(class_id)
    if not students:
        print("   No students to delete")
        return True
    
    print(f"   Found {len(students)} students:")
    for student in students:
        print(f"   - {student['name']} ({student['student_id']})")
    
    # Confirm deletion
    confirm = input(f"\n⚠️  Are you sure you want to delete ALL {len(students)} students? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Operation cancelled")
        return False
    
    # Double confirm
    confirm2 = input("⚠️  This action cannot be undone. Type 'DELETE' to confirm: ")
    if confirm2 != 'DELETE':
        print("❌ Operation cancelled")
        return False
    
    success = firebase.clear_class_data(class_id)
    if success:
        print("✅ All student data cleared successfully")
    return success

def main():
    parser = argparse.ArgumentParser(description='Manage student data in Firebase')
    parser.add_argument('--class-id', required=True, help='Class ID to manage')
    parser.add_argument('--action', choices=['list', 'delete', 'clear'], required=True,
                       help='Action to perform')
    parser.add_argument('--student-id', help='Student ID (for delete action)')
    
    args = parser.parse_args()
    
    # Initialize Firebase
    try:
        firebase = SimpleFirebase()
    except Exception as e:
        print(f"❌ Failed to connect to Firebase: {e}")
        return
    
    # Perform action
    if args.action == 'list':
        list_students(firebase, args.class_id)
        
    elif args.action == 'delete':
        if not args.student_id:
            print("❌ --student-id required for delete action")
            return
        delete_student(firebase, args.class_id, args.student_id)
        
    elif args.action == 'clear':
        clear_class(firebase, args.class_id)

if __name__ == "__main__":
    main()
