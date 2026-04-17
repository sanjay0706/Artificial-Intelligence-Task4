import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Tracker } from '@/src/lib/tracker';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Activity, BarChart3, Download, RefreshCw, FileVideo, Shield, Info, Settings } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export default function ResultsPage() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string>('');
  const [objectsCount, setObjectsCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);
  const [modelBase, setModelBase] = useState<'lite_mobilenet_v2' | 'mobilenet_v1' | 'mobilenet_v2'>('lite_mobilenet_v2');
  const [fps, setFps] = useState(0);
  const [detectionLog, setDetectionLog] = useState<{id: number, class: string, time: string}[]>([]);
  
  const trackerRef = useRef(new Tracker());
  const lastTimeRef = useRef(performance.now());
  const totalCountRef = useRef(totalCount);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    totalCountRef.current = totalCount;
  }, [totalCount]);

  useEffect(() => {
    const storedUrl = sessionStorage.getItem('processedVideoUrl');
    const storedName = sessionStorage.getItem('processedVideoName');
    if (storedUrl) {
      setVideoUrl(storedUrl);
      setVideoName(storedName || 'video.mp4');
    } else {
      navigate('/upload');
    }
    
    const loadModel = async () => {
      setIsLoading(true);
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
    loadModel();
  }, [navigate, modelBase]);

  useEffect(() => {
    let animationFrameId: number;
    const process = async () => {
      if (model && videoRef.current && canvasRef.current && isProcessing) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx && video.readyState === 4) {
          if (canvas.width !== video.videoWidth) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          const predictions = await model.detect(video);
          const filtered = predictions.filter(p => p.score >= 0.5).map(p => ({
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
            const label = `${obj.class} #${obj.id}`;
            ctx.fillRect(x, y - 25, ctx.measureText(label).width + 10, 25);
            ctx.fillStyle = '#44FF44';
            ctx.font = '12px JetBrains Mono';
            ctx.fillText(label, x + 5, y - 8);
          });
        }
      }
      animationFrameId = requestAnimationFrame(process);
    };
    if (isProcessing) process();
    return () => cancelAnimationFrame(animationFrameId);
  }, [model, isProcessing]);

  const handleDownload = () => {
    // In a real app, we'd use MediaRecorder to save the canvas stream
    // For now, we'll just show a message
    alert("In a production environment, this would download the processed video with all overlays.");
  };

  return (
    <div className="min-h-[calc(100vh-64px)] p-4 md:p-8 flex flex-col items-center gap-6">
      <header className="w-full max-w-5xl flex justify-between items-end border-b border-white/10 pb-4">
        <div>
          <h2 className="text-2xl font-bold uppercase tracking-tighter italic text-white">Neural Analysis Results</h2>
          <p className="mono-label">Processed: {videoName}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => navigate('/upload')}
            className="border-white/10 text-white/60 hover:bg-white/5 uppercase text-[10px] font-bold h-8"
          >
            <RefreshCw className="w-3 h-3 mr-2" />
            New Analysis
          </Button>
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
            {isLoading && <div className="text-white/40 mono-label animate-pulse">Loading Neural Engine...</div>}
            {videoUrl && (
              <video 
                ref={videoRef} 
                src={videoUrl} 
                className="absolute inset-0 w-full h-full object-cover opacity-60" 
                loop 
                muted 
                onPlay={() => setIsProcessing(true)}
                onLoadedMetadata={(e) => e.currentTarget.play()}
              />
            )}
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover z-10" />
          </div>
          <div className="p-4 border-t border-white/5 flex justify-between items-center">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-[#44FF44]" />
                <span className="text-[10px] font-bold uppercase text-white/60 tracking-widest">Neural Pass Complete</span>
              </div>
            </div>
            <Button 
              onClick={handleDownload}
              className="bg-white text-black hover:bg-white/90 font-bold uppercase text-[10px] tracking-widest h-8"
            >
              <Download className="w-3 h-3 mr-2" />
              Download Video
            </Button>
          </div>
        </Card>

        <div className="flex flex-col gap-6">
          <Card className="hardware-card border-none p-6 space-y-6">
            <div className="flex items-center gap-2 border-b border-white/5 pb-4">
              <Settings className="w-4 h-4 text-white/40" />
              <h3 className="text-sm font-bold uppercase tracking-widest">Neural Config</h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between text-xs font-bold">
                <span className="text-white/40 uppercase">Architecture</span>
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
          </Card>

          <Card className="hardware-card border-none p-6 space-y-6">
            <div className="flex items-center gap-2 border-b border-white/5 pb-4">
              <BarChart3 className="w-4 h-4 text-white/40" />
              <h3 className="text-sm font-bold uppercase tracking-widest">Analytics</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white/5 p-4 rounded-lg border border-white/5">
                <p className="text-[10px] font-bold uppercase text-white/30 mb-1">Total IDs</p>
                <p className="text-2xl font-black text-[#44FF44]">{totalCount}</p>
              </div>
              <div className="bg-white/5 p-4 rounded-lg border border-white/5">
                <p className="text-[10px] font-bold uppercase text-white/30 mb-1">Active</p>
                <p className="text-2xl font-black text-[#44FF44]">{objectsCount}</p>
              </div>
            </div>
          </Card>

          <Card className="hardware-card border-none p-6 space-y-4">
            <div className="flex items-center gap-2 border-b border-white/5 pb-4">
              <Activity className="w-4 h-4 text-white/40" />
              <h3 className="text-sm font-bold uppercase tracking-widest">Detection Log</h3>
            </div>
            <div className="space-y-2 max-h-[250px] overflow-y-auto custom-scrollbar pr-2">
              <AnimatePresence initial={false}>
                {detectionLog.map(log => (
                  <motion.div key={`${log.id}-${log.time}`} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="flex justify-between items-center text-[10px] bg-white/5 p-2 rounded border border-white/5">
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
