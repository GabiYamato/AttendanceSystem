import { initializeApp } from 'firebase/app'
import { getFirestore } from 'firebase/firestore'
import { getAuth } from 'firebase/auth'

const firebaseConfig = {
  apiKey: "AIzaSyBqJ8H_YourActualApiKey", // Replace with your actual API key from Firebase Console
  authDomain: "internfinall.firebaseapp.com",
  projectId: "internfinall",
  storageBucket: "internfinall.appspot.com",  
  messagingSenderId: "123456789", // Replace with your actual sender ID
  appId: "1:123456789:web:abcdef123456" // Replace with your actual app ID
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)

// Initialize Firebase services
export const db = getFirestore(app)
export const auth = getAuth(app)

export default app
