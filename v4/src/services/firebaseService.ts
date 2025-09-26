import { collection, addDoc, getDocs, doc, setDoc, query, orderBy, Timestamp, deleteDoc } from 'firebase/firestore'
import { db } from '../firebase'

export interface RegisteredFace {
  id: string
  name: string
  descriptor: number[] // Convert Float32Array to regular array for Firestore
  timestamp: string
}

export interface AttendanceRecord {
  id: string
  name: string
  timestamp: string
  status: 'present' | 'absent'
}

// Save registered face to Firebase
export const saveRegisteredFace = async (face: RegisteredFace) => {
  try {
    await setDoc(doc(db, 'registeredFaces', face.id), {
      ...face,
      createdAt: Timestamp.now()
    })
    console.log('Face saved to Firebase:', face.name)
  } catch (error) {
    console.error('Error saving face to Firebase:', error)
    throw error
  }
}

// Get all registered faces from Firebase
export const getRegisteredFaces = async (): Promise<RegisteredFace[]> => {
  try {
    const querySnapshot = await getDocs(collection(db, 'registeredFaces'))
    const faces: RegisteredFace[] = []
    
    querySnapshot.forEach((doc) => {
      const data = doc.data()
      faces.push({
        id: doc.id,
        name: data.name,
        descriptor: data.descriptor,
        timestamp: data.timestamp
      })
    })
    
    console.log('Loaded', faces.length, 'faces from Firebase')
    return faces
  } catch (error) {
    console.error('Error loading faces from Firebase:', error)
    return []
  }
}

// Save attendance record to Firebase
export const saveAttendanceRecord = async (record: AttendanceRecord) => {
  try {
    await addDoc(collection(db, 'attendanceRecords'), {
      ...record,
      createdAt: Timestamp.now()
    })
    console.log('Attendance record saved to Firebase:', record.name)
  } catch (error) {
    console.error('Error saving attendance record to Firebase:', error)
    throw error
  }
}

// Get all attendance records from Firebase
export const getAttendanceRecords = async (): Promise<AttendanceRecord[]> => {
  try {
    const q = query(collection(db, 'attendanceRecords'), orderBy('createdAt', 'desc'))
    const querySnapshot = await getDocs(q)
    const records: AttendanceRecord[] = []
    
    querySnapshot.forEach((doc) => {
      const data = doc.data()
      records.push({
        id: doc.id,
        name: data.name,
        timestamp: data.timestamp,
        status: data.status
      })
    })
    
    console.log('Loaded', records.length, 'attendance records from Firebase')
    return records
  } catch (error) {
    console.error('Error loading attendance records from Firebase:', error)
    return []
  }
}

// Clear all data from Firebase
export const clearAllFirebaseData = async () => {
  try {
    // Note: In a production app, you'd want to use batch operations
    // This is a simplified version
    console.log('Clearing all Firebase data...')
    
    // Get all faces and delete them
    const facesSnapshot = await getDocs(collection(db, 'registeredFaces'))
    const facePromises = facesSnapshot.docs.map(doc => deleteDoc(doc.ref))
    
    // Get all records and delete them  
    const recordsSnapshot = await getDocs(collection(db, 'attendanceRecords'))
    const recordPromises = recordsSnapshot.docs.map(doc => deleteDoc(doc.ref))
    
    await Promise.all([...facePromises, ...recordPromises])
    console.log('All Firebase data cleared')
  } catch (error) {
    console.error('Error clearing Firebase data:', error)
    throw error
  }
}
