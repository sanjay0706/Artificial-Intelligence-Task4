import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Tracker, Detection } from '@/src/lib/tracker';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Camera, Settings, Activity, Shield, Info, RefreshCw, Zap, ZapOff, Video, VideoOff, FileJson } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export default function WebcamPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [isFlashOn, setIsFlashOn] = useState(false);
  const [hasFlash, setHasFlash] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [fps, setFps] = useState(0);
  const [objectsCount, setObjectsCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);
  const [modelBase, setModelBase] = useState<'lite_mobilenet_v2' | 'mobilenet_v1' | 'mobilenet_v2'>('lite_mobilenet_v2');
  const [detectionLog, setDetectionLog] = useState<{uid: string, id: number, class: string, time: string}[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  
  const trackerRef = useRef(new Tracker());
  const lastTimeRef = useRef(performance.now());
  const thresholdRef = useRef(threshold);
  const totalCountRef = useRef(totalCount);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    thresholdRef.current = threshold;
  }, [threshold]);

  useEffect(() => {
    totalCountRef.current = totalCount;
  }, [totalCount]);

  const loadModel = async () => {
    setIsLoading(true);
    setError(null);
    try {
      await tf.ready();
      const loadedModel = await cocoSsd.load({ base: modelBase });
      setModel(loadedModel);
      setIsLoading(false);
    } catch (err) {
      setError("Neural Engine failed to initialize.");
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadModel();
  }, [modelBase]);

  useEffect(() => {
    if (model && !isDetecting && !error) {
      startCamera();
    }
  }, [model]);

  const startCamera = async (mode?: 'user' | 'environment') => {
    if (videoRef.current) {
      try {
        const targetMode = mode || facingMode;
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: targetMode, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        videoRef.current.srcObject = stream;
        const track = stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities() as any;
        setHasFlash(!!capabilities.torch);
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsDetecting(true);
        };
      } catch (err) {
        setError("Could not access camera.");
      }
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsDetecting(false);
    }
  };

  const toggleCamera = () => {
    const newMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newMode);
    stopCamera();
    setTimeout(() => startCamera(newMode), 50);
  };

  const capturePhoto = () => {
    if (canvasRef.current && videoRef.current) {
      const video = videoRef.current;
      const mainCanvas = canvasRef.current;
      
      if (video.readyState < 2) return;

      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      const ctx = captureCanvas.getContext('2d');
      
      if (ctx) {
        ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        ctx.drawImage(mainCanvas, 0, 0, captureCanvas.width, captureCanvas.height);
        
        try {
          captureCanvas.toBlob((blob) => {
            if (blob) {
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.download = `webcam-capture-${Date.now()}.png`;
              link.href = url;
              link.click();
              setTimeout(() => URL.revokeObjectURL(url), 100);
            }
          }, 'image/png');
        } catch (err) {
          console.error("Capture failed:", err);
          const dataUrl = captureCanvas.toDataURL('image/png');
          const link = document.createElement('a');
          link.download = `webcam-capture-${Date.now()}.png`;
          link.href = dataUrl;
          link.click();
        }
      }
    }
  };

  const toggleFlash = async () => {
    if (videoRef.current?.srcObject) {
      const track = (videoRef.current.srcObject as MediaStream).getVideoTracks()[0];
      try {
        await track.applyConstraints({ advanced: [{ torch: !isFlashOn }] } as any);
        setIsFlashOn(!isFlashOn);
      } catch (err) {}
    }
  };

  const startRecording = () => {
    if (canvasRef.current) {
      const stream = canvasRef.current.captureStream(30);
      const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      recordedChunksRef.current = [];
      recorder.ondataavailable = (e) => e.data.size > 0 && recordedChunksRef.current.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `webcam-track-${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
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
    a.download = `webcam-log-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    let animationFrameId: number;
    const detect = async () => {
      if (model && videoRef.current && canvasRef.current && isDetecting) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx && video.readyState === 4) {
          if (canvas.width !== video.videoWidth) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          const predictions = await model.detect(video);
          const currentThreshold = thresholdRef.current;
          const filtered = predictions.filter(p => p.score >= currentThreshold).map(p => ({
            bbox: p.bbox as [number, number, number, number],
            class: p.class,
            score: p.score
          }));
          const tracked = trackerRef.current.update(filtered);
          setObjectsCount(tracked.length);
          const newTotal = trackerRef.current.getTotalCount();
          const currentTotal = totalCountRef.current;
          if (newTotal > currentTotal) {
            const newTracks = tracked.filter(obj => obj.id > currentTotal);
            if (newTracks.length > 0) {
              setDetectionLog(prev => [
                ...newTracks.map(t => ({
                  uid: crypto.randomUUID(),
                  id: t.id,
                  class: t.class,
                  time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
                })),
                ...prev
              ].slice(0, 10));
            }
            setTotalCount(newTotal);
          }
          const now = performance.now();
          setFps(Math.round(1000 / (now - lastTimeRef.current)));
          lastTimeRef.current = now;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          tracked.forEach(obj => {
            const [x, y, w, h] = obj.bbox;
            ctx.strokeStyle = '#44FF44';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = 'rgba(21, 22, 25, 0.8)';
            const label = `${obj.class} #${obj.id} (${Math.round(obj.score * 100)}%)`;
            ctx.fillRect(x, y - 25, ctx.measureText(label).width + 10, 25);
            ctx.fillStyle = '#44FF44';
            ctx.font = '12px JetBrains Mono';
            ctx.fillText(label, x + 5, y - 8);
          });
        }
      }
      animationFrameId = requestAnimationFrame(detect);
    };
    if (isDetecting) detect();
    return () => cancelAnimationFrame(animationFrameId);
  }, [model, isDetecting]);

  return (
    <div className="min-h-[calc(100vh-64px)] p-4 md:p-8 flex flex-col items-center gap-6">
      <header className="w-full max-w-5xl flex justify-between items-end border-b border-white/10 pb-4">
        <div>
          <h2 className="text-2xl font-bold uppercase tracking-tighter italic text-white">Live Webcam Analysis</h2>
          <p className="mono-label">Real-time neural stream processing</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`status-dot ${isDetecting ? 'active' : 'inactive'}`}></span>
          <span className="text-xs font-bold uppercase text-white/60">{isDetecting ? 'Live' : 'Standby'}</span>
        </div>
      </header>

      <main className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 hardware-card overflow-hidden relative border-none">
          <div className="absolute top-4 left-4 z-10 flex gap-2">
            <Badge variant="outline" className="bg-black/50 text-[#44FF44] border-[#44FF44]/30 backdrop-blur-sm font-mono">FPS: {fps}</Badge>
            <Badge variant="outline" className="bg-black/50 text-[#44FF44] border-[#44FF44]/30 backdrop-blur-sm font-mono">OBJECTS: {objectsCount}</Badge>
            <Badge variant="outline" className="bg-black/50 text-[#44FF44] border-[#44FF44]/30 backdrop-blur-sm font-mono">TOTAL: {totalCount}</Badge>
          </div>
          <div className="relative aspect-video bg-black flex items-center justify-center">
            {isLoading && <div className="text-white/40 mono-label animate-pulse">Initializing Neural Engine...</div>}
            {error && <div className="text-red-500 mono-label">{error}</div>}
            <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover opacity-60" muted playsInline />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover z-10" />
          </div>
          <div className="p-4 border-t border-white/5 flex justify-between">
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={toggleCamera} className="border-white/10 text-white/60 hover:bg-white/5 uppercase text-[10px] font-bold h-8">
                <RefreshCw className="w-3 h-3 mr-2" /> {facingMode === 'user' ? 'Back' : 'Front'}
              </Button>
              <Button variant="outline" size="sm" onClick={capturePhoto} className="border-white/10 text-white/60 hover:bg-white/5 uppercase text-[10px] font-bold h-8">
                <Camera className="w-3 h-3 mr-2" /> Capture
              </Button>
              {hasFlash && isDetecting && (
                <Button variant="outline" size="sm" onClick={toggleFlash} className={`border-white/10 uppercase text-[10px] font-bold h-8 ${isFlashOn ? 'text-yellow-400 bg-yellow-400/10' : 'text-white/60'}`}>
                  {isFlashOn ? <Zap className="w-3 h-3 mr-2" /> : <ZapOff className="w-3 h-3 mr-2" />} Flash
                </Button>
              )}
              {isDetecting && (
                <Button variant="outline" size="sm" onClick={isRecording ? stopRecording : startRecording} className={`border-white/10 uppercase text-[10px] font-bold h-8 ${isRecording ? 'text-red-500 bg-red-500/10' : 'text-white/60'}`}>
                  {isRecording ? <VideoOff className="w-3 h-3 mr-2" /> : <Video className="w-3 h-3 mr-2" />} {isRecording ? 'Stop' : 'Record'}
                </Button>
              )}
            </div>
            {isDetecting && <Button variant="ghost" onClick={stopCamera} className="text-red-500 uppercase text-xs font-bold">Deactivate</Button>}
          </div>
        </Card>

        <div className="flex flex-col gap-6">
          <Card className="hardware-card border-none p-6 space-y-6">
            <div className="flex items-center gap-2 border-b border-white/5 pb-4">
              <Settings className="w-4 h-4 text-white/40" />
              <h3 className="text-sm font-bold uppercase tracking-widest">Parameters</h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between text-xs font-bold">
                <span className="text-white/40 uppercase">Model Architecture</span>
                <span className="text-[#44FF44]">{modelBase === 'lite_mobilenet_v2' ? 'Lite' : modelBase === 'mobilenet_v1' ? 'V1' : 'V2'}</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {(['lite_mobilenet_v2', 'mobilenet_v1', 'mobilenet_v2'] as const).map((base) => (
                  <Button
                    key={base}
                    variant="outline"
                    size="sm"
                    onClick={() => setModelBase(base)}
                    className={`h-7 text-[9px] font-bold uppercase ${modelBase === base ? 'bg-[#44FF44] text-black border-[#44FF44]' : 'text-white/40 border-white/10'}`}
                  >
                    {base === 'lite_mobilenet_v2' ? 'Lite' : base === 'mobilenet_v1' ? 'V1' : 'V2'}
                  </Button>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between text-xs font-bold">
                <span className="text-white/40 uppercase">Confidence</span>
                <span className="text-[#44FF44]">{Math.round(threshold * 100)}%</span>
              </div>
              <Slider value={[threshold * 100]} onValueChange={(v) => setThreshold(v[0] / 100)} max={100} step={1} />
            </div>
          </Card>

          <Card className="hardware-card border-none p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-4">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-white/40" />
                <h3 className="text-sm font-bold uppercase tracking-widest">Detection Log</h3>
              </div>
              {detectionLog.length > 0 && <Button variant="ghost" size="sm" onClick={exportLog} className="h-6 text-[10px] text-white/40 uppercase font-bold"><FileJson className="w-3 h-3 mr-1" /> Export</Button>}
            </div>
            <div className="space-y-2 max-h-[200px] overflow-y-auto custom-scrollbar pr-2">
              <AnimatePresence initial={false}>
                {detectionLog.map(log => (
                  <motion.div key={log.uid} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="flex justify-between items-center text-[10px] bg-white/5 p-2 rounded border border-white/5">
                    <span className="text-[#44FF44] font-bold">#{log.id} {log.class.toUpperCase()}</span>
                    <span className="text-white/30 font-mono">{log.time}</span>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
}
