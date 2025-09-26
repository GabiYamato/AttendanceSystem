import { useState, useEffect } from 'react'
import * as faceapi from 'face-api.js'

export const useFaceAPI = () => {
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      console.log('Loading Face API models...')
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models')
      ])
      setIsModelLoaded(true)
      setModelError(null)
      console.log('Face API models loaded successfully')
    } catch (error) {
      console.error('Error loading models:', error)
      setModelError('Error loading face recognition models')
      setIsModelLoaded(false)
    }
  }

  return { isModelLoaded, modelError, reloadModels: loadModels }
}
