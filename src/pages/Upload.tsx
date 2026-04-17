import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileVideo, Shield, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import { motion } from 'motion/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.type.startsWith('video/')) {
        setFile(selectedFile);
        setStatus('idle');
      } else {
        alert('Please select a valid video file.');
      }
    }
  };

  const handleUpload = () => {
    if (!file) return;
    
    setIsUploading(true);
    setStatus('uploading');
    
    // Simulate upload/processing delay
    setTimeout(() => {
      setStatus('success');
      setIsUploading(false);
      
      // Store the file URL in session storage to pass to results page
      const videoUrl = URL.createObjectURL(file);
      sessionStorage.setItem('processedVideoUrl', videoUrl);
      sessionStorage.setItem('processedVideoName', file.name);
      
      // Redirect to results after a short delay
      setTimeout(() => {
        navigate('/results');
      }, 1500);
    }, 2000);
  };

  return (
    <div className="min-h-[calc(100vh-64px)] flex flex-col items-center py-12 px-4">
      <div className="max-w-2xl w-full space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold uppercase tracking-tighter italic text-white">Upload Video</h2>
          <p className="mono-label">Prepare footage for neural analysis</p>
        </div>

        <Card className="hardware-card border-none p-12 flex flex-col items-center gap-8">
          {!file ? (
            <div 
              onClick={() => fileInputRef.current?.click()}
              className="w-full aspect-video rounded-2xl border-2 border-dashed border-white/10 hover:border-[#44FF44]/30 hover:bg-[#44FF44]/5 transition-all cursor-pointer flex flex-col items-center justify-center gap-4 group"
            >
              <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform">
                <Upload className="w-8 h-8 text-white/40 group-hover:text-[#44FF44]" />
              </div>
              <div className="text-center">
                <p className="text-sm font-bold uppercase tracking-widest text-white/60 group-hover:text-white">Click to browse</p>
                <p className="text-[10px] text-white/20 mt-1 uppercase">Supports MP4, AVI, WEBM</p>
              </div>
            </div>
          ) : (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="w-full space-y-6"
            >
              <div className="flex items-center gap-4 p-4 bg-white/5 rounded-xl border border-white/10">
                <div className="w-12 h-12 rounded-lg bg-[#44FF44]/10 flex items-center justify-center">
                  <FileVideo className="w-6 h-6 text-[#44FF44]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-bold text-white truncate">{file.name}</p>
                  <p className="text-[10px] text-white/40 uppercase">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setFile(null)}
                  disabled={isUploading}
                  className="text-red-500 hover:text-red-400 hover:bg-red-500/10 uppercase text-[10px] font-bold"
                >
                  Remove
                </Button>
              </div>

              {status === 'uploading' && (
                <div className="space-y-2">
                  <div className="flex justify-between text-[10px] uppercase font-bold">
                    <span className="text-white/40">Processing Stream</span>
                    <span className="text-[#44FF44]">Neural Engine Active</span>
                  </div>
                  <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                    <motion.div 
                      className="bg-[#44FF44] h-full"
                      initial={{ width: 0 }}
                      animate={{ width: '100%' }}
                      transition={{ duration: 2 }}
                    />
                  </div>
                </div>
              )}

              {status === 'success' && (
                <div className="flex items-center gap-2 text-[#44FF44] justify-center p-4 bg-[#44FF44]/10 rounded-lg border border-[#44FF44]/20">
                  <CheckCircle2 className="w-4 h-4" />
                  <span className="text-xs font-bold uppercase tracking-widest">Analysis Complete. Redirecting...</span>
                </div>
              )}

              <Button 
                onClick={handleUpload}
                disabled={isUploading || status === 'success'}
                className="w-full bg-[#44FF44] text-black hover:bg-[#33CC33] font-bold uppercase tracking-widest h-12"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Start Neural Analysis'
                )}
              </Button>
            </motion.div>
          )}

          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept="video/*" 
            className="hidden" 
          />
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3 p-4 bg-white/5 rounded-xl border border-white/5">
            <Shield className="w-5 h-5 text-white/20 mt-0.5" />
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase text-white/60">Local Processing</p>
              <p className="text-[10px] text-white/30 leading-relaxed">Your video is processed entirely on your device. No data is uploaded to external servers.</p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-white/5 rounded-xl border border-white/5">
            <AlertCircle className="w-5 h-5 text-white/20 mt-0.5" />
            <div className="space-y-1">
              <p className="text-[10px] font-bold uppercase text-white/60">Format Support</p>
              <p className="text-[10px] text-white/30 leading-relaxed">Optimized for H.264/AVC encoding. Large files may require higher system memory.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
