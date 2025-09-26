# Face Recognition Attendance System

A modern web-based attendance tracking system using facial recognition technology built with React, TypeScript, and face-api.js.

## Features

- **Face Registration**: Register users with facial data for recognition
- **Live Attendance Scanning**: Real-time face detection and attendance marking via webcam
- **Firebase Integration**: Cloud storage for attendance records and user data
- **Adjustable Recognition Threshold**: Customize confidence levels for attendance marking
- **Session Management**: Track attendance per session with clear statistics
- **Responsive UI**: Clean, user-friendly interface for desktop and mobile

## Technologies Used

- **Frontend**: React 18, TypeScript, Vite
- **Face Recognition**: face-api.js (TensorFlow.js)
- **Backend**: Firebase (Firestore, Authentication)
- **Styling**: CSS Modules
- **Build Tool**: Vite

## Installation

1. **Clone the repository**:
   ```bash
   git clone repo_url
   cd AttendanceSystem/v4
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up Firebase**:
   - Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
   - Enable Firestore and Authentication
   - Copy your Firebase config to `src/firebase.ts`

4. **Download face-api.js models**:
   - Models are already included in `public/models/`
   - If needed, download from [face-api.js GitHub](https://github.com/justadudewhohacks/face-api.js/)

5. **Start the development server**:
   ```bash
   npm run dev
   ```

## Usage

1. **Login**: Access the system with your credentials
2. **Register Faces**: Add new users by capturing their facial data
3. **Live Scanning**: Start the camera to detect and record attendance in real-time
4. **View Records**: Check attendance history on the dashboard

## Project Structure

```
v4/
├── public/
│   ├── models/          # face-api.js model files
│   └── vite.svg
├── src/
│   ├── components/      # React components
│   │   ├── AttendanceSystem.tsx
│   │   ├── Dashboard.tsx
│   │   ├── LiveAttendance.tsx
│   │   ├── LoginPage.tsx
│   │   └── RegistrationPage.tsx
│   ├── hooks/           # Custom React hooks
│   │   ├── useCamera.ts
│   │   └── useFaceAPI.ts
│   ├── services/        # Firebase services
│   │   └── firebaseService.ts
│   ├── types/           # TypeScript type definitions
│   │   └── index.ts
│   ├── App.tsx
│   ├── firebase.ts      # Firebase configuration
│   └── main.tsx
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Scripts

- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run preview`: Preview production build
- `npm run lint`: Run ESLint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Notes

- Ensure camera permissions are granted for live scanning
- Models load asynchronously; wait for loading to complete before scanning
- Adjust recognition threshold
