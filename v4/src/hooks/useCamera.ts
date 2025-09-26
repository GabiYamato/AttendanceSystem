import { useState, useRef, useCallback } from 'react'

export const useCamera = () => {
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const startCamera = useCallback(async () => {
    try {
      setCameraError(null)
      console.log('Starting camera...')
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setCameraActive(true)
        console.log('Camera started successfully')
      }
      return true
    } catch (error) {
      console.error('Error accessing camera:', error)
      setCameraError('Error accessing camera. Please check permissions.')
      setCameraActive(false)
      return false
    }
  }, [])

  const stopCamera = useCallback(() => {
    console.log('Stopping camera...')
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop()
        console.log('Camera track stopped')
      })
      streamRef.current = null
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      }
    }
    
    setCameraActive(false)
  }, [])

  const setupCanvas = useCallback(() => {
    return new Promise<void>((resolve) => {
      const video = videoRef.current
      const canvas = canvasRef.current
      
      if (!video || !canvas) {
        resolve()
        return
      }

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        console.log(`Canvas set to ${canvas.width}x${canvas.height}`)
        resolve()
      } else {
        const handleLoadedMetadata = () => {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          console.log(`Canvas set to ${canvas.width}x${canvas.height}`)
          video.removeEventListener('loadedmetadata', handleLoadedMetadata)
          resolve()
        }
        video.addEventListener('loadedmetadata', handleLoadedMetadata)
      }
    })
  }, [])

  return {
    cameraActive,
    cameraError,
    videoRef,
    canvasRef,
    startCamera,
    stopCamera,
    setupCanvas
  }
}
