export interface RegisteredFace {
  id: string
  name: string
  descriptor: Float32Array
  timestamp: string
}

export interface AttendanceRecord {
  id: string
  name: string
  timestamp: string
  status: 'present' | 'absent'
}
