import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Tracker, TrackedObject, Detection } from '@/src/lib/tracker';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Camera, Settings, Activity, Shield, Info, RefreshCw, Zap, ZapOff, Upload, Download, Video, VideoOff, FileJson, UploadCloud, CheckCircle2, AlertCircle, Volume2, VolumeX, Trash2, Circle, Square, Maximize2, Minimize2, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { io, Socket } from 'socket.io-client';
import { analyzeDetectionLog } from '@/src/services/geminiService';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  ResponsiveContainer, AreaChart, Area, ScatterChart, Scatter, ZAxis
} from 'recharts';

interface CollapsibleSectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  isOpen: boolean;
  onToggle: () => void;
  badge?: React.ReactNode;
}

const CollapsibleSection = ({ title, icon, children, isOpen, onToggle, badge }: CollapsibleSectionProps) => {
  return (
    <div className="border-b border-white/5 last:border-none">
      <button 
        onClick={onToggle}
        className="w-full flex items-center justify-between py-4 hover:bg-white/[0.02] transition-colors group px-6"
      >
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg transition-colors ${isOpen ? 'bg-[#44FF44]/10 text-[#44FF44]' : 'bg-white/5 text-white/40 group-hover:text-white/60'}`}>
            {icon}
          </div>
          <span className={`text-xs font-bold uppercase tracking-widest transition-colors ${isOpen ? 'text-white' : 'text-white/40 group-hover:text-white/60'}`}>
            {title}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {badge}
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            <ChevronDown className={`w-4 h-4 ${isOpen ? 'text-[#44FF44]' : 'text-white/20'}`} />
          </motion.div>
        </div>
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="pb-6 pt-2 space-y-6 px-6">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default function VisionDashboard() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [sourceType, setSourceType] = useState<'webcam' | 'file'>('webcam');
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [isFlashOn, setIsFlashOn] = useState(false);
  const [hasFlash, setHasFlash] = useState(false);
  const [threshold, setThreshold] = useState(() => {
    const saved = localStorage.getItem('vt_threshold');
    return saved ? parseFloat(saved) : 0.5;
  });
  const [maxInactiveFrames, setMaxInactiveFrames] = useState(() => {
    const saved = localStorage.getItem('vt_max_inactive');
    return saved ? parseInt(saved) : 30;
  });
  const [maxDistance, setMaxDistance] = useState(() => {
    const saved = localStorage.getItem('vt_max_distance');
    return saved ? parseInt(saved) : 100;
  });
  const [fps, setFps] = useState(0);
  const [objectsCount, setObjectsCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);
  const [detectionLog, setDetectionLog] = useState<{uid: string, id: number, class: string, time: string}[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isWebRTCActive, setIsWebRTCActive] = useState(false);
  const [webrtcStatus, setWebrtcStatus] = useState<string>('Disconnected');
  const [webrtcError, setWebrtcError] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [remoteDetections, setRemoteDetections] = useState<Record<string, { objects: number, fps: number, timestamp: number }>>({});
  
  const socketRef = useRef<Socket | null>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const trackerRef = useRef(new Tracker());
  const lastTimeRef = useRef(performance.now());
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [error, setError] = useState<string | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisType, setAnalysisType] = useState<'summary' | 'anomalies' | 'motion' | 'custom'>('summary');
  const [customPrompt, setCustomPrompt] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [exposure, setExposure] = useState(0);
  const [capabilities, setCapabilities] = useState<any>(null);
  const defaultSettingsRef = useRef<any>({});
  const [fpsSmoothing, setFpsSmoothing] = useState(0.5);
  const [resolution, setResolution] = useState<'480p' | '720p' | '1080p'>('720p');
  const [processingScale, setProcessingScale] = useState(1.0);
  const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    parameters: true,
    input: false,
    camera: false,
    security: false,
    ai: true,
    telemetry: false
  });

  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [selectedObjectId, setSelectedObjectId] = useState<number | null>(null);
  const historicalTracksRef = useRef<Record<number, TrackedObject>>({});

  const handleObjectSelect = (id: number) => {
    setSelectedObjectId(id);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const canvasAspect = canvas.width / canvas.height;
    const containerAspect = rect.width / rect.height;

    let x, y;
    if (containerAspect > canvasAspect) {
      const scaledHeight = rect.width / canvasAspect;
      const offsetY = (scaledHeight - rect.height) / 2;
      x = (e.clientX - rect.left) / rect.width * canvas.width;
      y = (e.clientY - rect.top + offsetY) / scaledHeight * canvas.height;
    } else {
      const scaledWidth = rect.height * canvasAspect;
      const offsetX = (scaledWidth - rect.width) / 2;
      x = (e.clientX - rect.left + offsetX) / scaledWidth * canvas.width;
      y = (e.clientY - rect.top) / rect.height * canvas.height;
    }

    const tracks = trackerRef.current.getTracks().filter(t => t.inactiveFrames === 0);
    const clickedObj = tracks.find(obj => {
      const [bx, by, bw, bh] = obj.bbox;
      return x >= bx && x <= bx + bw && y >= by && y <= by + bh;
    });

    if (clickedObj) {
      handleObjectSelect(clickedObj.id);
    }
  };

  const toggleSection = (id: string) => {
    setOpenSections(prev => ({ ...prev, [id]: !prev[id] }));
  };

  // Save threshold to local storage when it changes
  useEffect(() => {
    localStorage.setItem('vt_threshold', threshold.toString());
  }, [threshold]);

  // Update tracker parameters when they change and save to local storage
  useEffect(() => {
    trackerRef.current.setMaxInactiveFrames(maxInactiveFrames);
    trackerRef.current.setMaxDistance(maxDistance);
    localStorage.setItem('vt_max_inactive', maxInactiveFrames.toString());
    localStorage.setItem('vt_max_distance', maxDistance.toString());
  }, [maxInactiveFrames, maxDistance]);

  useEffect(() => {
    const applyCameraSettings = async () => {
      if (videoRef.current && videoRef.current.srcObject && isDetecting && sourceType === 'webcam') {
        const stream = videoRef.current.srcObject as MediaStream;
        const track = stream.getVideoTracks()[0];
        try {
          const advancedConstraints: any = {};
          
          if (capabilities?.brightness && brightness !== undefined) advancedConstraints.brightness = brightness;
          if (capabilities?.contrast && contrast !== undefined) advancedConstraints.contrast = contrast;
          if (capabilities?.exposureCompensation && exposure !== undefined) advancedConstraints.exposureCompensation = exposure;
          
          if (Object.keys(advancedConstraints).length > 0) {
            await track.applyConstraints({
              advanced: [advancedConstraints]
            } as any);
          }
        } catch (err) {
          console.error("Error applying camera settings:", err);
        }
      }
    };
    applyCameraSettings();
  }, [brightness, contrast, exposure, isDetecting, capabilities, sourceType]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
    } else {
      setRecordingDuration(0);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const toggleFullScreen = () => {
    const element = document.getElementById('main-video-container');
    if (!element) return;

    if (!document.fullscreenElement) {
      element.requestFullscreen().then(() => setIsFullScreen(true)).catch(err => {
        console.error(`Error attempting to enable full-screen mode: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
      setIsFullScreen(false);
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullScreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  const loadModel = async () => {
    setIsLoading(true);
    setError(null);
    setIsSimulating(false);
    
    console.log("Initializing Neural Engine...");
    
    if (!navigator.onLine) {
      setError("System is offline. Please check your network connection.");
      setIsLoading(false);
      return;
    }

    try {
      await tf.ready();
      console.log("TensorFlow.js ready. Backend:", tf.getBackend());
      
      const modelBases: cocoSsd.ObjectDetectionBaseModel[] = [
        'lite_mobilenet_v2',
        'mobilenet_v2',
        'mobilenet_v1'
      ];

      for (const base of modelBases) {
        try {
          console.log(`Attempting to fetch model: ${base}...`);
          const loadedModel = await cocoSsd.load({ base });
          setModel(loadedModel);
          setIsLoading(false);
          console.log(`Successfully loaded model: ${base}`);
          return;
        } catch (e) {
          console.warn(`Failed to fetch ${base}:`, e);
        }
      }

      throw new Error("All model variants failed to fetch. This usually indicates that Google Storage (storage.googleapis.com) is unreachable from your current network.");
    } catch (err) {
      console.error("Neural Engine Initialization Error:", err);
      setError(err instanceof Error ? err.message : "Neural Engine failed to initialize. This may be due to network restrictions or a temporary service outage.");
      setIsLoading(false);
    }
  };

  const startSimulation = () => {
    setIsSimulating(true);
    setError(null);
    setIsDetecting(true);
  };

  useEffect(() => {
    if (!isLoading && !error && !model && !isSimulating) {
      loadModel();
    }
  }, []);

  // Auto-start camera when model is ready
  useEffect(() => {
    if (model && !isDetecting && !error) {
      startCamera();
    }
  }, [model]);

  const startCamera = async (mode?: 'user' | 'environment', res?: '480p' | '720p' | '1080p') => {
    stopCamera();
    setSourceType('webcam');
    const targetMode = mode || facingMode;
    const targetRes = res || resolution;
    
    const resMap = {
      '480p': { width: 640, height: 480 },
      '720p': { width: 1280, height: 720 },
      '1080p': { width: 1920, height: 1080 }
    };

    if (videoRef.current) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: targetMode, 
            width: { ideal: resMap[targetRes].width }, 
            height: { ideal: resMap[targetRes].height } 
          },
          audio: false,
        });
        
        videoRef.current.srcObject = stream;
        
        // Check for capabilities
        const track = stream.getVideoTracks()[0];
        const caps = track.getCapabilities() as any;
        setCapabilities(caps);
        setHasFlash(!!caps.torch);

        // Initialize state from current settings
        const settings = track.getSettings() as any;
        defaultSettingsRef.current = {
          brightness: settings.brightness,
          contrast: settings.contrast,
          exposureCompensation: settings.exposureCompensation
        };
        
        if (settings.brightness !== undefined) setBrightness(settings.brightness);
        if (settings.contrast !== undefined) setContrast(settings.contrast);
        if (settings.exposureCompensation !== undefined) setExposure(settings.exposureCompensation);

        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsDetecting(true);
          initWebRTC(stream);
        };
      } catch (err) {
        console.error("Camera error:", err);
        setError("Could not access camera. Please ensure you have granted permissions.");
      }
    }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && videoRef.current) {
      stopCamera();
      setSourceType('file');
      setHasFlash(false);
      setIsFlashOn(false);
      setVideoName(file.name);
      
      // Chunked Upload Logic
      setIsUploading(true);
      setUploadProgress(0);
      
      const CHUNK_SIZE = 1024 * 1024 * 2; // 2MB chunks
      const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
      const fileName = `${Date.now()}-${file.name.replace(/\s+/g, '_')}`;

      try {
        for (let i = 0; i < totalChunks; i++) {
          const start = i * CHUNK_SIZE;
          const end = Math.min(start + CHUNK_SIZE, file.size);
          const chunk = file.slice(start, end);

          const formData = new FormData();
          formData.append("chunk", chunk);
          formData.append("fileName", fileName);
          formData.append("chunkIndex", i.toString());
          formData.append("totalChunks", totalChunks.toString());

          // Robust retry logic for each chunk
          let retries = 3;
          let success = false;
          while (retries > 0 && !success) {
            try {
              const response = await fetch("/api/upload/chunk", {
                method: "POST",
                body: formData,
              });
              if (!response.ok) throw new Error(`Server responded with ${response.status}`);
              success = true;
            } catch (chunkErr) {
              retries--;
              if (retries === 0) throw chunkErr;
              console.warn(`Chunk ${i} upload failed, retrying... (${retries} retries left)`);
              await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s before retry
            }
          }

          setUploadProgress(Math.round(((i + 1) / totalChunks) * 100));
        }

        const completeRes = await fetch("/api/upload/complete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fileName }),
        });

        const { url } = await completeRes.json();
        
        setVideoUrl(url);
        setIsProcessing(true);
        
        videoRef.current.src = url;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsDetecting(true);
        };
      } catch (err) {
        console.error("Upload error:", err);
        setError("Failed to upload video. Falling back to local playback.");
        
        // Fallback to local blob URL if server upload fails
        const url = URL.createObjectURL(file);
        setVideoUrl(url);
        setIsProcessing(true);
        videoRef.current.src = url;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsDetecting(true);
        };
      } finally {
        setIsUploading(false);
      }
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const toggleCamera = async () => {
    const newMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newMode);
    if (isDetecting) {
      // Small delay to ensure state update and stream release
      setTimeout(() => startCamera(newMode), 50);
    }
  };

  const changeResolution = (newRes: '480p' | '720p' | '1080p') => {
    setResolution(newRes);
    if (isDetecting && sourceType === 'webcam') {
      startCamera(facingMode, newRes);
    }
  };

  const capturePhoto = () => {
    if (canvasRef.current && videoRef.current) {
      const video = videoRef.current;
      const mainCanvas = canvasRef.current;
      
      // Ensure video is ready
      if (video.readyState < 2) return;

      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      const ctx = captureCanvas.getContext('2d');
      
      if (ctx) {
        // Draw the video frame
        ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        
        // Draw the detections if any
        // Since mainCanvas is already sized to videoWidth/Height in the detection loop,
        // we can draw it directly.
        ctx.drawImage(mainCanvas, 0, 0, captureCanvas.width, captureCanvas.height);
        
        try {
          captureCanvas.toBlob((blob) => {
            if (blob) {
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.download = `vision-capture-${Date.now()}.png`;
              link.href = url;
              link.click();
              
              // Cleanup
              setTimeout(() => URL.revokeObjectURL(url), 100);
            }
          }, 'image/png');
        } catch (err) {
          console.error("Capture failed:", err);
          // Fallback to data URL if toBlob fails (e.g. in very old browsers)
          const dataUrl = captureCanvas.toDataURL('image/png');
          const link = document.createElement('a');
          link.download = `vision-capture-${Date.now()}.png`;
          link.href = dataUrl;
          link.click();
        }
      }
    }
  };

  const toggleFlash = async () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      const track = stream.getVideoTracks()[0];
      try {
        await track.applyConstraints({
          advanced: [{ torch: !isFlashOn }]
        } as any);
        setIsFlashOn(!isFlashOn);
      } catch (err) {
        console.error("Flash error:", err);
      }
    }
  };

  const stopCamera = () => {
    stopWebRTC();
    if (videoRef.current) {
      if (videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      } else if (videoRef.current.src) {
        URL.revokeObjectURL(videoRef.current.src);
        videoRef.current.src = '';
        setVideoUrl(null);
        setVideoName('');
        setIsProcessing(false);
      }
      setIsDetecting(false);
    }
  };

  const resetTracker = () => {
    trackerRef.current.reset();
    setTotalCount(0);
    setObjectsCount(0);
    setDetectionLog([]);
  };

  const clearLog = () => {
    setDetectionLog([]);
    setAiAnalysis(null);
  };

  const resetCameraSettings = () => {
    const defaults = defaultSettingsRef.current;
    if (defaults.brightness !== undefined) setBrightness(defaults.brightness);
    if (defaults.contrast !== undefined) setContrast(defaults.contrast);
    if (defaults.exposureCompensation !== undefined) setExposure(defaults.exposureCompensation);
  };

  const startRecording = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      
      // Try to find a supported mime type
      const mimeTypes = [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm',
        'video/mp4'
      ];
      
      let selectedMimeType = '';
      for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          selectedMimeType = type;
          break;
        }
      }

      const recorder = new MediaRecorder(stream, selectedMimeType ? { mimeType: selectedMimeType } : undefined);
      
      recordedChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };
      
      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: selectedMimeType || 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const extension = selectedMimeType.includes('mp4') ? 'mp4' : 'webm';
        a.download = `vision-raw-capture-${new Date().getTime()}.${extension}`;
        a.click();
        URL.revokeObjectURL(url);
      };
      
      recorder.start(1000); // Capture in 1s chunks
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const exportLog = () => {
    const data = JSON.stringify(detectionLog, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detection-log-${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const runAiAnalysis = async () => {
    if (detectionLog.length === 0) return;
    setIsAnalyzing(true);
    setAiAnalysis(null);
    try {
      const result = await analyzeDetectionLog(detectionLog, analysisType, customPrompt);
      setAiAnalysis(result);
    } catch (err) {
      console.error("AI Analysis Error:", err);
      setAiAnalysis("Analysis failed. Ensure GEMINI_API_KEY is configured.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const startVideoAnalysis = async () => {
    if (!videoRef.current || !model || sourceType !== 'file' || isAnalyzingVideo) return;
    
    setIsAnalyzingVideo(true);
    setAnalysisProgress(0);
    setAiAnalysis(null);
    resetTracker();
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    video.pause();
    const duration = video.duration;
    const step = 0.5; // Process every 0.5s for efficiency
    
    try {
      for (let time = 0; time <= duration; time += step) {
        if (!isAnalyzingVideo) break;
        video.currentTime = time;
        
        // Wait for seek
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject('Seek timeout'), 5000);
          const onSeeked = () => {
            clearTimeout(timeout);
            video.removeEventListener('seeked', onSeeked);
            resolve(null);
          };
          video.addEventListener('seeked', onSeeked);
        });

        // Perform detection
        let input: any = video;
        if (processingScale < 1.0) {
          if (!processingCanvasRef.current) processingCanvasRef.current = document.createElement('canvas');
          const pCanvas = processingCanvasRef.current;
          const pCtx = pCanvas.getContext('2d');
          if (pCtx) {
            pCanvas.width = video.videoWidth * processingScale;
            pCanvas.height = video.videoHeight * processingScale;
            pCtx.drawImage(video, 0, 0, pCanvas.width, pCanvas.height);
            input = pCanvas;
          }
        }

        const predictions = await model.detect(input);
        const filtered = predictions
          .filter(p => p.score >= threshold)
          .map(p => {
            const scale = 1 / processingScale;
            return {
              bbox: [p.bbox[0] * scale, p.bbox[1] * scale, p.bbox[2] * scale, p.bbox[3] * scale] as [number, number, number, number],
              class: p.class,
              score: p.score
            };
          });

        const tracked = trackerRef.current.update(filtered);
        setObjectsCount(tracked.length);
        
        // Draw the detections on the main UI canvas for feedback
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        drawNeuralOverlay(ctx, tracked, canvas.width, canvas.height);

        const newTotal = trackerRef.current.getTotalCount();
        if (newTotal > totalCount) {
          const newTracks = tracked.filter(obj => obj.id > totalCount);
          if (newTracks.length > 0) {
            setDetectionLog(prev => [
              ...newTracks.map(t => ({
                uid: crypto.randomUUID(),
                id: t.id,
                class: t.class,
                time: `SCAN @ ${time.toFixed(1)}s`
              })),
              ...prev
            ].slice(0, 50));
            setTotalCount(newTotal);
          }
        }

        setAnalysisProgress(Math.round((time / duration) * 100));
        
        // Brief pause to allow UI update
        await new Promise(r => setTimeout(r, 10));
      }

      setAnalysisProgress(100);
      setTimeout(() => {
        setIsAnalyzingVideo(false);
        runAiAnalysis();
      }, 1000);
    } catch (err) {
      console.error("Deep Scan Error:", err);
      setError("Video analysis interrupted.");
      setIsAnalyzingVideo(false);
    }
  };

  const initWebRTC = async (stream: MediaStream) => {
    setWebrtcStatus('Connecting...');
    setWebrtcError(null);
    
    try {
      // Initialize Socket.io for signaling
      const socket = io({
        reconnectionAttempts: 5,
        timeout: 10000,
      });
      socketRef.current = socket;

      socket.on('connect_error', (err) => {
        console.error('Socket connection error:', err);
        setWebrtcError('Signaling server unreachable');
        setWebrtcStatus('Failed');
      });

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });
      peerConnectionRef.current = pc;

      // Add tracks to peer connection
      stream.getTracks().forEach(track => pc.addTrack(track, stream));

      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate) {
          socket.emit('ice-candidate', { candidate: event.candidate });
        }
      };

      pc.oniceconnectionstatechange = () => {
        console.log('ICE Connection State:', pc.iceConnectionState);
        if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
          setWebrtcError('ICE connection failed or dropped');
          setIsWebRTCActive(false);
        }
      };

      pc.onconnectionstatechange = () => {
        setWebrtcStatus(pc.connectionState);
        console.log('Connection State:', pc.connectionState);
        
        if (pc.connectionState === 'connected') {
          setIsWebRTCActive(true);
          setWebrtcError(null);
        } else if (pc.connectionState === 'failed') {
          setWebrtcError('Peer connection failed');
          setIsWebRTCActive(false);
        } else if (pc.connectionState === 'closed') {
          setIsWebRTCActive(false);
        }
      };

      // Signaling listeners
      socket.on('answer', async (data) => {
        try {
          await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
        } catch (err) {
          console.error('Error setting remote description:', err);
          setWebrtcError('Failed to establish peer handshake');
        }
      });

      socket.on('ice-candidate', async (data) => {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
        } catch (e) {
          console.error('Error adding ice candidate', e);
        }
      });

      socket.on('remote-detection', (data) => {
        setRemoteDetections(prev => ({
          ...prev,
          [data.clientId]: {
            objects: data.objects,
            fps: data.fps,
            timestamp: Date.now()
          }
        }));
      });

      // Create offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      socket.emit('offer', { offer });
    } catch (err) {
      console.error('WebRTC Init Error:', err);
      setWebrtcError('Failed to initialize WebRTC');
      setWebrtcStatus('Error');
    }
  };

  const retryWebRTC = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      initWebRTC(videoRef.current.srcObject as MediaStream);
    }
  };

  const playAlert = () => {
    if (!isAudioEnabled) return;
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();

      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
      oscillator.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.1);

      gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.1);

      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      oscillator.start();
      oscillator.stop(audioCtx.currentTime + 0.1);
      
      // Close context after play to free resources
      setTimeout(() => audioCtx.close(), 200);
    } catch (e) {
      console.error("Audio alert failed:", e);
    }
  };

  const stopWebRTC = () => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    setIsWebRTCActive(false);
    setWebrtcStatus('Disconnected');
  };

  useEffect(() => {
    let animationFrameId: number;

    const detect = async () => {
      if ((model || isSimulating) && videoRef.current && canvasRef.current && isDetecting && !isAnalyzingVideo) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (ctx && video.readyState === 4) {
          // Set canvas size to match video
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          let filteredDetections: Detection[] = [];

          if (isSimulating) {
            // Generate mock detections for demo purposes if model fails
            const time = performance.now() / 1000;
            filteredDetections = [
              {
                bbox: [
                  100 + Math.sin(time) * 50, 
                  100 + Math.cos(time) * 30, 
                  120, 180
                ],
                class: 'person',
                score: 0.95
              },
              {
                bbox: [
                  300 + Math.cos(time * 0.8) * 40, 
                  200 + Math.sin(time * 1.2) * 20, 
                  80, 60
                ],
                class: 'laptop',
                score: 0.88
              }
            ];
          } else if (model) {
            // Detect objects
            let input: any = video;
            
            // Apply processing scale for file uploads to improve performance
            if (sourceType === 'file' && processingScale < 1.0) {
              if (!processingCanvasRef.current) {
                processingCanvasRef.current = document.createElement('canvas');
              }
              const pCanvas = processingCanvasRef.current;
              const pCtx = pCanvas.getContext('2d');
              if (pCtx) {
                pCanvas.width = video.videoWidth * processingScale;
                pCanvas.height = video.videoHeight * processingScale;
                pCtx.drawImage(video, 0, 0, pCanvas.width, pCanvas.height);
                input = pCanvas;
              }
            }

            const predictions = await model.detect(input);
            filteredDetections = predictions
              .filter(p => p.score >= threshold)
              .map(p => {
                // If we scaled the input, we must scale the bounding boxes back
                const scale = sourceType === 'file' ? 1 / processingScale : 1;
                return {
                  bbox: [
                    p.bbox[0] * scale,
                    p.bbox[1] * scale,
                    p.bbox[2] * scale,
                    p.bbox[3] * scale
                  ] as [number, number, number, number],
                  class: p.class,
                  score: p.score
                };
              });
          }

          // Track objects
          const trackedObjects = trackerRef.current.update(filteredDetections);
          setObjectsCount(trackedObjects.length);
          
          // Update historical tracks for details view
          trackedObjects.forEach(obj => {
            historicalTracksRef.current[obj.id] = { ...obj };
          });
          
          const newTotal = trackerRef.current.getTotalCount();
          if (newTotal > totalCount) {
            // New objects detected, add to log
            const newTracks = trackedObjects.filter(obj => obj.id > totalCount);
            if (newTracks.length > 0) {
              playAlert();
              setDetectionLog(prev => [
                ...newTracks.map(t => ({
                  uid: crypto.randomUUID(),
                  id: t.id,
                  class: t.class,
                  time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
                })),
                ...prev
              ].slice(0, 50));
            }
            setTotalCount(newTotal);
          }

          // Calculate FPS
          const now = performance.now();
          const delta = now - lastTimeRef.current;
          lastTimeRef.current = now;
          const currentFps = 1000 / delta;
          setFps(prev => {
            const smoothed = prev * fpsSmoothing + currentFps * (1 - fpsSmoothing);
            return Math.round(smoothed);
          });

          // Emit detection data to server for remote monitoring
          if (socketRef.current && socketRef.current.connected && Math.random() > 0.9) {
            socketRef.current.emit('detection-update', {
              objects: trackedObjects.length,
              fps: currentFps
            });
          }

          drawNeuralOverlay(ctx, trackedObjects, canvas.width, canvas.height);
        }
      }
      animationFrameId = requestAnimationFrame(detect);
    };

    if (isDetecting) {
      detect();
    }

    return () => cancelAnimationFrame(animationFrameId);
  }, [model, isDetecting, threshold, isSimulating, fpsSmoothing, sourceType, processingScale, isAnalyzingVideo, totalCount]);

  const drawNeuralOverlay = (ctx: CanvasRenderingContext2D, trackedObjects: TrackedObject[], width: number, height: number) => {
    // Draw results
    ctx.clearRect(0, 0, width, height);
    
    // Draw scanning line
    const scanY = (performance.now() / 20) % height;
    ctx.strokeStyle = 'rgba(68, 255, 68, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, scanY);
    ctx.lineTo(width, scanY);
    ctx.stroke();

    // Draw center crosshair
    const centerX = width / 2;
    const centerY = height / 2;
    ctx.strokeStyle = 'rgba(68, 255, 68, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX - 20, centerY);
    ctx.lineTo(centerX + 20, centerY);
    ctx.moveTo(centerX, centerY - 20);
    ctx.lineTo(centerX, centerY + 20);
    ctx.stroke();

    trackedObjects.forEach(obj => {
      const [x, y, boxWidth, boxHeight] = obj.bbox;
      const isSelected = obj.id === selectedObjectId;
      
      // Draw path trail on canvas if selected
      if (isSelected && obj.history.length > 1) {
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.moveTo(obj.history[0].centroid[0], obj.history[0].centroid[1]);
        for (let i = 1; i < obj.history.length; i++) {
          ctx.lineTo(obj.history[i].centroid[0], obj.history[i].centroid[1]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw box glow
      ctx.shadowBlur = isSelected ? 20 : 10;
      ctx.shadowColor = isSelected ? '#FFFFFF' : '#44FF44';
      
      // Draw box
      ctx.strokeStyle = isSelected ? '#FFFFFF' : '#44FF44';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x, y, boxWidth, boxHeight);

      // Draw corner brackets
      const bracketSize = 15;
      ctx.lineWidth = 4;
      ctx.strokeStyle = isSelected ? '#FFFFFF' : '#44FF44';
      ctx.beginPath();
      ctx.moveTo(x, y + bracketSize);
      ctx.lineTo(x, y);
      ctx.lineTo(x + bracketSize, y);
      ctx.stroke();

      // Top Right
      ctx.beginPath();
      ctx.moveTo(x + boxWidth - bracketSize, y);
      ctx.lineTo(x + boxWidth, y);
      ctx.lineTo(x + boxWidth, y + bracketSize);
      ctx.stroke();

      // Bottom Left
      ctx.beginPath();
      ctx.moveTo(x, y + boxHeight - bracketSize);
      ctx.lineTo(x, y + boxHeight);
      ctx.lineTo(x + bracketSize, y + boxHeight);
      ctx.stroke();

      // Bottom Right
      ctx.beginPath();
      ctx.moveTo(x + boxWidth - bracketSize, y + boxHeight);
      ctx.lineTo(x + boxWidth, y + boxHeight);
      ctx.lineTo(x + boxWidth, y + boxHeight - bracketSize);
      ctx.stroke();

      ctx.shadowBlur = 0; // Reset shadow

      // Draw label background
      ctx.fillStyle = isSelected ? 'rgba(255, 255, 255, 0.95)' : 'rgba(0, 0, 0, 0.85)';
      const label = `${obj.class.toUpperCase()} // ID:${obj.id}`;
      const confidence = `${Math.round(obj.score * 100)}%`;
      
      ctx.font = 'bold 10px JetBrains Mono';
      const labelWidth = ctx.measureText(label).width;
      const confWidth = ctx.measureText(confidence).width;
      const totalWidth = Math.max(labelWidth, confWidth) + 15;
      
      ctx.fillRect(x, y - 35, totalWidth, 35);
      ctx.strokeStyle = isSelected ? '#000000' : '#44FF44';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y - 35, totalWidth, 35);

      // Draw label text
      ctx.fillStyle = isSelected ? '#000000' : '#44FF44';
      ctx.fillText(label, x + 7, y - 22);
      ctx.fillStyle = isSelected ? 'rgba(0, 0, 0, 0.6)' : 'rgba(68, 255, 68, 0.6)';
      ctx.fillText(`CONFIDENCE: ${confidence}`, x + 7, y - 8);

      // Draw ID point
      ctx.fillStyle = '#44FF44';
      ctx.beginPath();
      ctx.arc(obj.centroid[0], obj.centroid[1], 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col items-center gap-6">
      <header className="w-full max-w-5xl flex justify-between items-end border-b border-black/10 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter uppercase italic font-serif">VisionTrack AI</h1>
          <p className="mono-label">Autonomous Object Detection & Tracking System // v1.0.4</p>
        </div>
        <div className="flex gap-8">
          <div className="flex flex-col items-end">
            <span className="mono-label">Active Tracks</span>
            <span className="text-xl font-black text-[#44FF44] italic leading-none">{objectsCount}</span>
          </div>
          <div className="flex flex-col items-end">
            <span className="mono-label">Total Session</span>
            <span className="text-xl font-black text-[#44FF44] italic leading-none">{totalCount}</span>
          </div>
          <div className="flex flex-col items-end">
            <span className="mono-label">System Status</span>
            <div className="flex items-center gap-2">
              <span className={`status-dot ${isDetecting ? 'active' : 'inactive'}`}></span>
              <span className="text-xs font-bold uppercase tracking-widest">{isDetecting ? 'Live' : 'Standby'}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Viewport */}
        <Card className="lg:col-span-2 hardware-card overflow-hidden relative border-none">
          <div className="absolute top-4 left-4 z-10 flex gap-2">
            <Badge variant="outline" className="bg-black/50 text-[#44FF44] border-[#44FF44]/30 backdrop-blur-sm font-mono">
              OBJECTS: {objectsCount}
            </Badge>
            <Badge variant="outline" className="bg-black/50 text-[#44FF44] border-[#44FF44]/30 backdrop-blur-sm font-mono">
              TOTAL: {totalCount}
            </Badge>
          </div>

          <div className="absolute top-4 right-4 z-10 flex flex-col items-end">
            <div className="bg-black/60 backdrop-blur-md border border-[#44FF44]/20 rounded-lg p-3 flex flex-col items-end shadow-[0_0_20px_rgba(68,255,68,0.1)]">
              <div className="flex items-center gap-2 mb-1">
                <div className={`w-1.5 h-1.5 rounded-full bg-[#44FF44] ${fps > 0 ? 'animate-pulse shadow-[0_0_8px_#44FF44]' : 'opacity-20'}`} />
                <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest">Real-time Performance</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-3xl font-black text-[#44FF44] font-mono leading-none">{fps}</span>
                <span className="text-[10px] font-bold text-[#44FF44]/60 uppercase">FPS</span>
              </div>
              <div className="w-full bg-white/5 h-1 mt-2 rounded-full overflow-hidden">
                <motion.div 
                  className="bg-[#44FF44] h-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, (fps / 60) * 100)}%` }}
                  transition={{ type: "spring", stiffness: 100 }}
                />
              </div>
            </div>
          </div>

          <div id="main-video-container" className={`relative bg-black flex items-center justify-center overflow-hidden ${isFullScreen ? 'fixed inset-0 z-[100]' : 'aspect-video rounded-2xl shadow-2xl'}`}>
            {isLoading && (
              <div className="flex flex-col items-center gap-4 text-white/50">
                <Activity className="w-12 h-12 animate-pulse" />
                <p className="mono-label">Initializing Neural Engine...</p>
              </div>
            )}

            {error && (
              <div className="flex flex-col items-center gap-4 text-red-400 p-6 text-center">
                <Shield className="w-12 h-12 opacity-50" />
                <div className="space-y-2">
                  <p className="mono-label text-red-500">Critical Error</p>
                  <p className="text-xs max-w-xs">{error}</p>
                </div>
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    onClick={loadModel}
                    className="border-red-500/50 text-red-500 hover:bg-red-500/10 uppercase text-xs font-bold"
                  >
                    Retry
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={startSimulation}
                    className="border-white/20 text-white/60 hover:bg-white/5 uppercase text-xs font-bold"
                  >
                    Run Simulation
                  </Button>
                </div>
              </div>
            )}
            
            <video
              ref={videoRef}
              className={`absolute inset-0 w-full h-full object-cover ${error ? 'hidden' : ''}`}
              muted
              playsInline
              loop={sourceType === 'file'}
            />
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className={`absolute inset-0 w-full h-full object-cover z-10 cursor-crosshair ${error ? 'hidden' : ''}`}
            />

            {/* Recording Controls */}
            <div className="absolute top-4 left-4 z-40 flex items-center gap-3">
              {isRecording ? (
                <motion.div 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center gap-3 bg-black/60 backdrop-blur-md border border-red-500/30 px-3 py-1.5 rounded-full"
                >
                  <div className="relative">
                    <Circle className="w-3 h-3 text-red-500 fill-red-500 animate-pulse" />
                    <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-20" />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-white uppercase tracking-widest leading-none">Recording</span>
                    <span className="text-[9px] font-mono text-red-500/80 leading-none mt-0.5">
                      {Math.floor(recordingDuration / 60).toString().padStart(2, '0')}:
                      {(recordingDuration % 60).toString().padStart(2, '0')}
                    </span>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={stopRecording}
                    className="h-6 w-6 p-0 hover:bg-red-500/20 text-red-500"
                  >
                    <Square className="w-3 h-3 fill-red-500" />
                  </Button>
                </motion.div>
              ) : (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={startRecording}
                  disabled={!isDetecting || isUploading}
                  className="bg-black/40 backdrop-blur-md border-white/10 text-white/60 hover:text-white hover:bg-black/60 rounded-full px-4 h-8 uppercase text-[10px] font-bold tracking-widest group"
                >
                  <Circle className="w-3 h-3 mr-2 text-red-500 group-hover:fill-red-500 transition-colors" />
                  Start Recording
                </Button>
              )}
            </div>

            {/* Full Screen Toggle */}
            <div className="absolute top-4 right-4 z-40">
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={toggleFullScreen}
                className="h-8 w-8 p-0 bg-black/40 backdrop-blur-md border border-white/10 text-white/60 hover:text-white hover:bg-black/60 rounded-full"
              >
                {isFullScreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </Button>
            </div>

            {isUploading && (
              <div className="absolute inset-0 z-30 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center p-12">
                <div className="w-full max-w-md space-y-6">
                  <div className="flex flex-col items-center gap-4">
                    <div className="relative">
                      <UploadCloud className="w-12 h-12 text-[#44FF44] animate-pulse" />
                      <div className="absolute -top-1 -right-1">
                        <motion.div 
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        >
                          <RefreshCw className="w-4 h-4 text-[#44FF44]/50" />
                        </motion.div>
                      </div>
                    </div>
                    <div className="text-center">
                      <h3 className="text-sm font-bold uppercase tracking-[0.2em] text-white mb-1">Neural Data Ingestion</h3>
                      <p className="text-[10px] text-white/40 uppercase font-mono">Transmitting: {videoName}</p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between items-end">
                      <span className="text-[10px] font-bold text-[#44FF44] uppercase tracking-widest">Progress</span>
                      <span className="text-xl font-black text-[#44FF44] italic">{uploadProgress}%</span>
                    </div>
                    <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden border border-white/5">
                      <motion.div 
                        className="h-full bg-gradient-to-r from-[#44FF44]/50 to-[#44FF44]"
                        initial={{ width: 0 }}
                        animate={{ width: `${uploadProgress}%` }}
                        transition={{ type: "spring", bounce: 0, duration: 0.5 }}
                      />
                    </div>
                    <div className="flex justify-between text-[8px] text-white/20 uppercase font-bold tracking-tighter">
                      <span>Byte Stream Active</span>
                      <span>Encrypted Tunnel</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {videoUrl && isProcessing && (
              <div className="absolute inset-0 z-20 bg-black flex flex-col">
                <div className="flex-1 relative">
                  <video 
                    src={videoUrl} 
                    className="w-full h-full object-contain"
                    controls
                    autoPlay
                    loop
                  />
                  <div className="absolute top-4 right-4">
                    <Button 
                      variant="secondary" 
                      size="sm" 
                      onClick={() => setIsProcessing(false)}
                      className="bg-black/50 text-white border-white/10 backdrop-blur-sm uppercase text-[10px] font-bold"
                    >
                      Close Player
                    </Button>
                  </div>
                </div>
                <div className="p-3 bg-white/5 border-t border-white/5 flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <FileJson className="w-4 h-4 text-[#44FF44]" />
                    <span className="text-[10px] font-bold uppercase text-white/60 tracking-widest">Playing: {videoName}</span>
                  </div>
                </div>
              </div>
            )}

            {!isDetecting && !isLoading && !error && (
              <div className="z-10 flex flex-col items-center gap-6 w-full max-w-md">
                {isUploading ? (
                  <div className="w-full space-y-4 text-center">
                    <div className="relative w-24 h-24 mx-auto">
                      <div className="absolute inset-0 rounded-full border-4 border-white/5" />
                      <motion.div 
                        className="absolute inset-0 rounded-full border-4 border-[#44FF44] border-t-transparent"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <UploadCloud className="w-8 h-8 text-[#44FF44]" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <h3 className="text-white font-bold uppercase tracking-widest text-sm">Uploading Assets</h3>
                      <p className="text-white/40 text-[10px] uppercase tracking-wider">{videoName}</p>
                    </div>
                    <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden border border-white/10">
                      <motion.div 
                        className="bg-[#44FF44] h-full shadow-[0_0_10px_#44FF44]"
                        initial={{ width: 0 }}
                        animate={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <div className="flex justify-between items-center text-[10px] font-mono text-white/40">
                      <span>{uploadProgress}% COMPLETE</span>
                      <span>CHUNKED TRANSFER ACTIVE</span>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="w-20 h-20 rounded-full border-2 border-dashed border-white/20 flex items-center justify-center">
                      <Camera className="w-8 h-8 text-white/40" />
                    </div>
                    <div className="flex flex-col sm:flex-row gap-4">
                      <Button 
                        onClick={startCamera}
                        className="bg-[#44FF44] text-black hover:bg-[#33CC33] font-bold uppercase tracking-widest px-8"
                      >
                        Activate Webcam
                      </Button>
                      <Button 
                        onClick={triggerFileUpload}
                        variant="outline"
                        className="border-white/20 text-white hover:bg-white/5 font-bold uppercase tracking-widest px-8"
                      >
                        <Upload className="w-4 h-4 mr-2" />
                        Upload Video
                      </Button>
                      <input 
                        type="file" 
                        ref={fileInputRef} 
                        onChange={handleFileChange} 
                        accept="video/*" 
                        className="hidden" 
                      />
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="p-4 border-t border-white/5 flex justify-between items-center">
            <div className="flex gap-2">
              {sourceType === 'webcam' && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={toggleCamera}
                  className="border-white/10 text-white/60 hover:bg-white/5 uppercase text-[10px] font-bold h-8"
                >
                  <RefreshCw className="w-3 h-3 mr-2" />
                  {facingMode === 'user' ? 'Back Cam' : 'Front Cam'}
                </Button>
              )}
              
              {hasFlash && isDetecting && sourceType === 'webcam' && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={toggleFlash}
                  className={`border-white/10 uppercase text-[10px] font-bold h-8 ${isFlashOn ? 'text-yellow-400 bg-yellow-400/10' : 'text-white/60 hover:bg-white/5'}`}
                >
                  {isFlashOn ? <Zap className="w-3 h-3 mr-2" /> : <ZapOff className="w-3 h-3 mr-2" />}
                  Flash
                </Button>
              )}

              {sourceType === 'webcam' && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={capturePhoto}
                  className="border-white/10 text-white/60 hover:bg-white/5 uppercase text-[10px] font-bold h-8"
                >
                  <Camera className="w-3 h-3 mr-2" />
                  Capture
                </Button>
              )}
              
              {isDetecting && (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`border-white/10 uppercase text-[10px] font-bold h-8 ${isRecording ? 'text-red-500 bg-red-500/10 border-red-500/30' : 'text-white/60 hover:bg-white/5'}`}
                >
                  {isRecording ? <VideoOff className="w-3 h-3 mr-2" /> : <Video className="w-3 h-3 mr-2" />}
                  {isRecording ? 'Stop Rec' : 'Record'}
                </Button>
              )}
            </div>
            {isDetecting && (
              <Button 
                variant="ghost" 
                onClick={stopCamera}
                className="text-red-500 hover:text-red-400 hover:bg-red-500/10 uppercase text-xs font-bold"
              >
                Deactivate
              </Button>
            )}
          </div>
        </Card>

        {/* Sidebar Controls */}
        <div className="flex flex-col gap-4 lg:col-span-1">
          <Card className="hardware-card border-none overflow-hidden flex flex-col">
            <CollapsibleSection
              title="Neural Parameters"
              icon={<Settings className="w-4 h-4" />}
              isOpen={openSections.parameters}
              onToggle={() => toggleSection('parameters')}
            >
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="mono-label">Confidence Threshold</span>
                  <div className="flex items-center gap-2">
                    <input 
                      type="number" 
                      value={Math.round(threshold * 100)}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 0;
                        const clamped = Math.min(Math.max(val, 0), 100);
                        setThreshold(clamped / 100);
                      }}
                      className="w-12 h-6 bg-white/5 border border-white/10 rounded text-[10px] font-bold text-[#44FF44] text-center focus:border-[#44FF44]/50 outline-none transition-colors"
                    />
                    <span className="text-[10px] font-bold text-white/20">%</span>
                  </div>
                </div>
                <Slider
                  value={[threshold * 100]}
                  onValueChange={(val) => {
                    const clamped = Math.min(Math.max(val[0], 0), 100);
                    setThreshold(clamped / 100);
                  }}
                  min={0}
                  max={100}
                  step={1}
                />
                <p className="text-[10px] text-white/30 leading-relaxed uppercase font-mono">
                  Minimum confidence score for detection. Higher = conservative.
                </p>
              </div>

              <div className="space-y-4 pt-4 border-t border-white/5">
                <div className="flex justify-between items-center">
                  <span className="mono-label">Max Inactive Frames</span>
                  <input 
                    type="number" 
                    value={maxInactiveFrames}
                    onChange={(e) => {
                      const val = parseInt(e.target.value) || 1;
                      const clamped = Math.min(Math.max(val, 1), 100);
                      setMaxInactiveFrames(clamped);
                    }}
                    className="w-12 h-6 bg-white/5 border border-white/10 rounded text-[10px] font-bold text-[#44FF44] text-center focus:border-[#44FF44]/50 outline-none transition-colors"
                  />
                </div>
                <Slider
                  value={[maxInactiveFrames]}
                  onValueChange={(val) => {
                    const clamped = Math.min(Math.max(val[0], 1), 100);
                    setMaxInactiveFrames(clamped);
                  }}
                  max={100}
                  min={1}
                  step={1}
                />
              </div>

              <div className="space-y-4 pt-4 border-t border-white/5">
                <div className="flex justify-between items-center">
                  <span className="mono-label">Max Match Distance</span>
                  <div className="flex items-center gap-2">
                    <input 
                      type="number" 
                      value={maxDistance}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 10;
                        const clamped = Math.min(Math.max(val, 10), 500);
                        setMaxDistance(clamped);
                      }}
                      className="w-12 h-6 bg-white/5 border border-white/10 rounded text-[10px] font-bold text-[#44FF44] text-center focus:border-[#44FF44]/50 outline-none transition-colors"
                    />
                    <span className="text-[10px] font-bold text-white/20">px</span>
                  </div>
                </div>
                <Slider
                  value={[maxDistance]}
                  onValueChange={(val) => {
                    const clamped = Math.min(Math.max(val[0], 10), 500);
                    setMaxDistance(clamped);
                  }}
                  max={500}
                  min={10}
                  step={5}
                />
              </div>

              <div className="space-y-4 pt-4 border-t border-white/5">
                <div className="flex justify-between items-center">
                  <span className="mono-label">FPS Smoothing</span>
                  <div className="flex items-center gap-2">
                    <input 
                      type="number" 
                      value={Math.round(fpsSmoothing * 100)}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 0;
                        const clamped = Math.min(Math.max(val, 0), 95);
                        setFpsSmoothing(clamped / 100);
                      }}
                      className="w-12 h-6 bg-white/5 border border-white/10 rounded text-[10px] font-bold text-[#44FF44] text-center focus:border-[#44FF44]/50 outline-none transition-colors"
                    />
                    <span className="text-[10px] font-bold text-white/20">%</span>
                  </div>
                </div>
                <Slider
                  value={[fpsSmoothing * 100]}
                  onValueChange={(val) => {
                    const clamped = Math.min(Math.max(val[0], 0), 95);
                    setFpsSmoothing(clamped / 100);
                  }}
                  max={95}
                  min={0}
                  step={5}
                />
              </div>
            </CollapsibleSection>

            <CollapsibleSection
              title="Input Engine"
              icon={<Upload className="w-4 h-4" />}
              isOpen={openSections.input}
              onToggle={() => toggleSection('input')}
              badge={
                <Badge variant="outline" className={`text-[9px] font-bold uppercase ${isWebRTCActive ? 'text-[#44FF44] border-[#44FF44]/30' : 'text-white/20'}`}>
                  {sourceType}
                </Badge>
              }
            >
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="mono-label">{sourceType === 'webcam' ? 'Resolution' : 'Scale'}</span>
                  <span className="text-xs font-bold text-[#44FF44]">{sourceType === 'webcam' ? resolution : `${Math.round(processingScale * 100)}%`}</span>
                </div>
                
                {sourceType === 'webcam' ? (
                  <div className="grid grid-cols-3 gap-2">
                    {(['480p', '720p', '1080p'] as const).map((res) => (
                      <Button
                        key={res}
                        variant="outline"
                        size="sm"
                        onClick={() => changeResolution(res)}
                        className={`h-7 text-[9px] font-bold uppercase ${resolution === res ? 'bg-[#44FF44] text-black border-[#44FF44]' : 'text-white/40 border-white/10'}`}
                      >
                        {res}
                      </Button>
                    ))}
                  </div>
                ) : (
                  <Slider
                    value={[processingScale * 100]}
                    onValueChange={(val) => setProcessingScale(val[0] / 100)}
                    max={100}
                    min={25}
                    step={25}
                  />
                )}
              </div>

              <div className="pt-4 border-t border-white/5">
                <span className="mono-label block mb-3">Switch Source</span>
                {isUploading ? (
                  <div className="space-y-2">
                    <div className="flex justify-between items-center text-[9px] font-bold uppercase text-[#44FF44]">
                      <span>Processing...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                      <motion.div 
                        className="bg-[#44FF44] h-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="flex gap-2">
                    <Button 
                      variant="outline"
                      size="sm"
                      onClick={startCamera}
                      className={`flex-1 uppercase text-[10px] font-bold transition-all h-8 ${sourceType === 'webcam' ? 'bg-[#44FF44] text-black border-[#44FF44]' : 'border-white/10 text-white/60 hover:bg-white/5'}`}
                    >
                      <Camera className="w-3 h-3 mr-2" />
                      Webcam
                    </Button>
                    <Button 
                      variant="outline"
                      size="sm"
                      onClick={triggerFileUpload}
                      className={`flex-1 uppercase text-[10px] font-bold transition-all h-8 ${sourceType === 'file' ? 'bg-[#44FF44] text-black border-[#44FF44]' : 'border-white/10 text-white/60 hover:bg-white/5'}`}
                    >
                      <Upload className="w-3 h-3 mr-2" />
                      File
                    </Button>
                  </div>
                )}
              </div>

              <div className="space-y-4 pt-4 border-t border-white/5">
                <div className="flex items-center justify-between">
                  <span className="mono-label">WebRTC Tunnel</span>
                  <Badge variant="outline" className={`text-[8px] font-bold uppercase ${isWebRTCActive ? 'text-[#44FF44] border-[#44FF44]/30' : webrtcError ? 'text-red-500 border-red-500/30' : 'text-white/20 border-white/10'}`}>
                    {webrtcStatus}
                  </Badge>
                </div>
                
                {webrtcError && (
                  <div className="bg-red-500/10 border border-red-500/20 p-2 rounded flex flex-col gap-2">
                    <p className="text-[10px] font-bold uppercase text-red-500 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" /> {webrtcError}
                    </p>
                    <Button variant="outline" size="sm" onClick={retryWebRTC} className="h-6 text-[9px] border-red-500/30 text-red-500 uppercase font-bold">Retry</Button>
                  </div>
                )}
              </div>
            </CollapsibleSection>

            {sourceType === 'webcam' && capabilities && (
              <CollapsibleSection
                title="Hardware Adjust"
                icon={<Camera className="w-4 h-4" />}
                isOpen={openSections.camera}
                onToggle={() => toggleSection('camera')}
                badge={
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={(e) => { e.stopPropagation(); resetCameraSettings(); }}
                    className="h-5 px-2 text-[8px] text-white/40 hover:text-white uppercase font-bold"
                  >
                    Reset
                  </Button>
                }
              >
                {capabilities.brightness && (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="mono-label">Brightness</span>
                      <span className="text-xs font-bold text-[#44FF44]">{brightness}</span>
                    </div>
                    <Slider
                      value={[brightness]}
                      onValueChange={(val) => setBrightness(val[0])}
                      max={capabilities.brightness.max}
                      min={capabilities.brightness.min}
                      step={capabilities.brightness.step || 1}
                    />
                  </div>
                )}

                {capabilities.contrast && (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="mono-label">Contrast</span>
                      <span className="text-xs font-bold text-[#44FF44]">{contrast}</span>
                    </div>
                    <Slider
                      value={[contrast]}
                      onValueChange={(val) => setContrast(val[0])}
                      max={capabilities.contrast.max}
                      min={capabilities.contrast.min}
                      step={capabilities.contrast.step || 1}
                    />
                  </div>
                )}

                {capabilities.exposureCompensation && (
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="mono-label">Exposure</span>
                      <span className="text-xs font-bold text-[#44FF44]">{exposure}</span>
                    </div>
                    <Slider
                      value={[exposure]}
                      onValueChange={(val) => setExposure(val[0])}
                      max={capabilities.exposureCompensation.max}
                      min={capabilities.exposureCompensation.min}
                      step={capabilities.exposureCompensation.step || 0.1}
                    />
                  </div>
                )}
              </CollapsibleSection>
            )}

            <CollapsibleSection
              title="System Status"
              icon={<Shield className="w-4 h-4" />}
              isOpen={openSections.security}
              onToggle={() => toggleSection('security')}
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="mono-label">Alert Audio</span>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => setIsAudioEnabled(!isAudioEnabled)}
                    className={`h-6 w-6 p-0 ${isAudioEnabled ? 'text-[#44FF44]' : 'text-white/20'}`}
                  >
                    {isAudioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                  </Button>
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-white/5 p-3 rounded-lg border border-white/5">
                    <span className="block mono-label mb-1 text-[8px]">Protocol</span>
                    <span className="text-[10px] font-bold text-[#44FF44]">SORT-JS</span>
                  </div>
                  <div className="bg-white/5 p-3 rounded-lg border border-white/5 text-right">
                    <span className="block mono-label mb-1 text-[8px]">Latent</span>
                    <span className="text-[10px] font-bold text-[#44FF44]">0.2ms</span>
                  </div>
                </div>

                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={resetTracker}
                  className="w-full text-white/40 border-white/10 hover:border-white/20 uppercase text-[10px] font-bold h-8"
                >
                  Hard Reset Tracker
                </Button>
              </div>
            </CollapsibleSection>

            <CollapsibleSection
              title="Neural Insights"
              icon={<Activity className="w-4 h-4" />}
              isOpen={openSections.ai}
              onToggle={() => toggleSection('ai')}
              badge={
                <Badge className="bg-[#44FF44] text-black text-[9px] font-bold h-4">
                  {detectionLog.length}
                </Badge>
              }
            >
              <div className="space-y-4">
                <div className="flex gap-2 bg-white/5 p-1 rounded-lg">
                  {(['summary', 'anomalies', 'motion'] as const).map((type) => (
                    <button
                      key={type}
                      onClick={() => setAnalysisType(type)}
                      className={`flex-1 py-1.5 text-[9px] font-bold uppercase rounded transition-all ${analysisType === type ? 'bg-[#44FF44] text-black' : 'text-white/40 hover:text-white/60'}`}
                    >
                      {type}
                    </button>
                  ))}
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center mb-1">
                    <span className="mono-label">Capture Log</span>
                    <div className="flex gap-1 items-center">
                      {sourceType === 'file' && videoUrl && !isAnalyzingVideo && (
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          onClick={startVideoAnalysis}
                          className="h-6 px-2 text-[#44FF44]/60 hover:text-[#44FF44] text-[10px] uppercase font-bold"
                        >
                          <RefreshCw className="w-3 h-3 mr-1" /> Deep Scan
                        </Button>
                      )}
                      <Button variant="ghost" size="sm" onClick={clearLog} disabled={detectionLog.length === 0} className="h-6 px-2 text-red-500/60 hover:text-red-500 text-[10px] uppercase font-bold">
                        <Trash2 className="w-3 h-3 mr-1" /> Clear
                      </Button>
                    </div>
                  </div>

                  {isAnalyzingVideo && (
                    <div className="space-y-2 bg-[#44FF44]/5 border border-[#44FF44]/10 p-3 rounded-lg animate-pulse">
                      <div className="flex justify-between items-center text-[10px] font-bold text-[#44FF44]">
                        <span className="uppercase tracking-widest">Neural Scan Active</span>
                        <span>{analysisProgress}%</span>
                      </div>
                      <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                        <motion.div 
                          className="bg-[#44FF44] h-full"
                          animate={{ width: `${analysisProgress}%` }}
                        />
                      </div>
                      <p className="text-[8px] text-white/30 uppercase text-center">Batch processing video frames...</p>
                    </div>
                  )}

                  <div className="flex justify-between items-center text-[9px] uppercase font-black tracking-widest text-[#44FF44]/60 px-4 py-2.5 bg-white/[0.05] rounded-t-lg border-x border-t border-white/10 shadow-[inner_0_1px_0_rgba(255,255,255,0.05)]">
                    <div className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-[#44FF44] rounded-full animate-pulse" />
                      <span>Neural Registry // Object Stream</span>
                    </div>
                    <span className="font-mono opacity-40">System_Time</span>
                  </div>
                  
                  <div className="space-y-1.5 max-h-[450px] overflow-y-auto custom-scrollbar pr-1 bg-black/40 rounded-b-lg border border-white/10 p-3 min-h-[150px] shadow-inner">
                    <AnimatePresence initial={false} mode="popLayout">
                      {detectionLog.length === 0 ? (
                        <div className="py-12 text-center">
                          <p className="text-[10px] text-white/10 uppercase tracking-[0.2em] font-mono animate-pulse">Awaiting neural input...</p>
                        </div>
                      ) : (
                        detectionLog.map((log) => (
                          <motion.div 
                            key={log.uid}
                            initial={{ opacity: 0, x: -10, filter: 'blur(5px)' }}
                            animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            layout
                            onClick={() => handleObjectSelect(log.id)}
                            className="flex justify-between items-center text-[10px] bg-white/[0.03] px-3 py-2.5 rounded border border-white/5 hover:bg-white/[0.08] transition-colors group cursor-pointer"
                          >
                            <div className="flex items-center gap-2">
                              <Circle className="w-1.5 h-1.5 text-[#44FF44] fill-[#44FF44]/20 group-hover:animate-ping" />
                              <span className="text-white/80 font-bold tracking-tight">
                                <span className="text-[#44FF44] opacity-50 mr-1">#{log.id}</span>
                                {log.class.toUpperCase()}
                              </span>
                            </div>
                            <span className="text-white/30 font-mono text-[9px] group-hover:text-[#44FF44]/60 transition-colors">{log.time}</span>
                          </motion.div>
                        ))
                      )}
                    </AnimatePresence>
                  </div>
                  {detectionLog.length > 5 && (
                    <div className="text-center">
                      <p className="text-[8px] text-white/10 uppercase tracking-widest">Showing latest 50 events</p>
                    </div>
                  )}
                </div>

                {detectionLog.length > 0 && (
                  <Button 
                    className="w-full bg-[#44FF44] hover:bg-[#44FF44]/90 text-black font-black uppercase text-xs h-10 tracking-widest shadow-[0_0_20px_rgba(68,255,68,0.2)]"
                    onClick={runAiAnalysis}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
                    {isAnalyzing ? 'Processing...' : 'Generate AI Report'}
                  </Button>
                )}

                <AnimatePresence>
                  {aiAnalysis && (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="bg-[#44FF44]/10 border border-[#44FF44]/20 p-4 rounded-lg relative"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-[10px] font-black uppercase text-[#44FF44] tracking-widest">Neural Report</span>
                        <button onClick={() => setAiAnalysis(null)} className="text-white/20 hover:text-white"><VolumeX className="w-3 h-3" /></button>
                      </div>
                      <p className="text-[11px] text-white/80 leading-relaxed font-mono italic">
                        "{aiAnalysis}"
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </CollapsibleSection>

            <CollapsibleSection
              title="Telemetry"
              icon={<Info className="w-4 h-4" />}
              isOpen={openSections.telemetry}
              onToggle={() => toggleSection('telemetry')}
            >
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-[10px] font-bold uppercase">
                    <span className="text-white/40">Neural Load</span>
                    <span className="text-[#44FF44]">{(fps > 0 ? (1000/fps).toFixed(1) : 0)}ms</span>
                  </div>
                  <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                    <motion.div 
                      className="bg-[#44FF44] h-full"
                      animate={{ width: `${Math.min(100, (1000/fps) * 2)}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <span className="text-[9px] text-white/20 uppercase font-bold">Memory</span>
                    <span className="text-xs font-mono block">142.4 MB</span>
                  </div>
                  <div className="space-y-1 text-right">
                    <span className="text-[9px] text-white/20 uppercase font-bold">Uptime</span>
                    <span className="text-xs font-mono block">04:12:09</span>
                  </div>
                </div>

                {Object.keys(remoteDetections).length > 0 && (
                  <div className="pt-4 border-t border-white/5 space-y-2">
                    <span className="text-[9px] text-white/20 uppercase font-bold block mb-1">Peer Nodes</span>
                    {Object.entries(remoteDetections).map(([id, data]: [string, any]) => (
                      <div key={id} className="flex justify-between items-center text-[10px] bg-white/5 p-2 rounded">
                        <span className="text-white/60">Node_{id.slice(0, 4)}</span>
                        <span className="text-[#44FF44]">{data.fps} FPS</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CollapsibleSection>
          </Card>
        </div>
      </main>

      <footer className="w-full max-w-5xl flex justify-center py-8">
        <p className="text-[10px] text-black/30 uppercase tracking-[0.2em] font-bold">
          &copy; 2026 VisionTrack Systems // All Rights Reserved
        </p>
      </footer>

      {/* Object Detail Modal */}
      <AnimatePresence>
        {selectedObjectId !== null && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedObjectId(null)}
              className="absolute inset-0 bg-black/80 backdrop-blur-md"
            />
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="relative w-full max-w-2xl hardware-card bg-[#0a0a0a] border-[#44FF44]/30 overflow-hidden"
            >
              {/* Modal Header */}
              <div className="p-6 border-b border-[#44FF44]/10 flex justify-between items-start bg-black/40">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <Badge className="bg-[#44FF44] text-black font-mono">ID:{selectedObjectId}</Badge>
                    <h3 className="text-xl font-black text-white uppercase italic">
                      {historicalTracksRef.current[selectedObjectId]?.class || 'Unknown Object'}
                    </h3>
                  </div>
                  <p className="text-[10px] text-[#44FF44]/50 font-mono uppercase tracking-widest">Neural Persistence Track // Global History</p>
                </div>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  onClick={() => setSelectedObjectId(null)}
                  className="text-white/20 hover:text-white hover:bg-white/5"
                >
                  <RefreshCw className="w-4 h-4 rotate-45" />
                </Button>
              </div>

              {/* Modal Content */}
              {historicalTracksRef.current[selectedObjectId] ? (
                <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Stats */}
                  <div className="space-y-6">
                    <div>
                      <span className="mono-label block mb-4">Centroid Path (X,Y)</span>
                      <div className="h-40 w-full bg-black/40 rounded-lg border border-white/5 p-4">
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                            <XAxis type="number" dataKey="x" hide domain={['auto', 'auto']} />
                            <YAxis type="number" dataKey="y" hide domain={['auto', 'auto']} reversed />
                            <ZAxis type="number" range={[10, 10]} />
                            <Scatter 
                              name="Path" 
                              data={historicalTracksRef.current[selectedObjectId].history.map((h, i) => ({
                                index: i,
                                x: Math.round(h.centroid[0]),
                                y: Math.round(h.centroid[1])
                              }))} 
                              fill="#44FF44" 
                              line={{ stroke: '#44FF44', strokeWidth: 1, strokeDasharray: '3 3' }} 
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div>
                      <span className="mono-label block mb-2">Confidence Registry</span>
                      <div className="h-24 w-full bg-black/40 rounded-lg border border-white/5 p-2">
                         <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={historicalTracksRef.current[selectedObjectId].history.map((h, i) => ({
                            index: i,
                            score: Math.round(h.score * 100)
                          }))}>
                            <defs>
                              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#44FF44" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#44FF44" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <Area type="monotone" dataKey="score" stroke="#44FF44" fillOpacity={1} fill="url(#scoreGradient)" isAnimationActive={false} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* Tracking Data */}
                  <div className="space-y-4">
                    <div className="bg-white/5 rounded-lg p-4 border border-white/5">
                      <span className="text-[9px] font-bold text-white/40 uppercase mb-3 block">Current Attributes</span>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-1">
                          <span className="text-[8px] text-white/20 uppercase">Last Score</span>
                          <div className="text-lg font-black text-[#44FF44] font-mono">
                            {Math.round(historicalTracksRef.current[selectedObjectId].score * 100)}%
                          </div>
                        </div>
                        <div className="space-y-1">
                          <span className="text-[8px] text-white/20 uppercase">Inactive Frames</span>
                          <div className="text-lg font-black text-white/60 font-mono">
                            {historicalTracksRef.current[selectedObjectId].inactiveFrames}
                          </div>
                        </div>
                        <div className="space-y-1">
                          <span className="text-[8px] text-white/20 uppercase">Centroid X</span>
                          <div className="text-lg font-black text-white/60 font-mono">
                            {Math.round(historicalTracksRef.current[selectedObjectId].centroid[0])}
                          </div>
                        </div>
                        <div className="space-y-1">
                          <span className="text-[8px] text-white/20 uppercase">Centroid Y</span>
                          <div className="text-lg font-black text-white/60 font-mono">
                            {Math.round(historicalTracksRef.current[selectedObjectId].centroid[1])}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <span className="text-[9px] font-bold text-white/40 uppercase block mb-2">Detection History (Last 5)</span>
                      <div className="space-y-1 max-h-[140px] overflow-y-auto pr-2 custom-scrollbar">
                        {historicalTracksRef.current[selectedObjectId].history.slice(-5).reverse().map((h, i) => (
                          <div key={i} className="flex justify-between items-center bg-white/[0.02] p-2 rounded border border-white/5 text-[9px] font-mono">
                            <span className="text-white/40">{new Date(h.timestamp).toLocaleTimeString()}</span>
                            <span className="text-[#44FF44]">{Math.round(h.score * 100)}% Match</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-12 text-center">
                  <p className="text-white/20 font-mono text-xs uppercase animate-pulse">Retrieving telemetry data...</p>
                </div>
              )}

              {/* Footer Decoration */}
              <div className="bg-[#44FF44]/5 px-6 py-2 flex justify-between items-center border-t border-[#44FF44]/10">
                 <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map(i => <div key={i} className="w-1 h-1 bg-[#44FF44]/30" />)}
                 </div>
                 <span className="text-[8px] text-[#44FF44]/40 font-mono uppercase tracking-[0.3em]">Neural Processor Active</span>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
