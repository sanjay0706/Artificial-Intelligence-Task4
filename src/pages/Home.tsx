import React from 'react';
import { Link } from 'react-router-dom';
import { Camera, Upload, BarChart3, Shield, Zap, Activity, Target } from 'lucide-react';
import { motion } from 'motion/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

export default function Home() {
  return (
    <div className="min-h-[calc(100vh-64px)] flex flex-col items-center py-12 px-4">
      {/* Hero Section */}
      <div className="max-w-4xl w-full text-center space-y-8 mb-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#44FF44]/10 border border-[#44FF44]/20 text-[#44FF44] text-[10px] font-bold uppercase tracking-[0.2em]"
        >
          <Zap className="w-3 h-3" />
          Neural Engine v1.0.4 Active
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-5xl md:text-7xl font-black tracking-tighter uppercase italic text-white"
        >
          Object Detection & <br />
          <span className="text-[#44FF44]">Tracking System</span>
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-white/40 max-w-2xl mx-auto text-sm md:text-base leading-relaxed"
        >
          A professional-grade computer vision suite designed for real-time analysis. 
          Leverage state-of-the-art neural networks to detect, classify, and track 
          multiple objects with persistent identity across frames.
        </motion.p>
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex flex-wrap justify-center gap-4 pt-4"
        >
          <Link to="/upload">
            <Button className="bg-white text-black hover:bg-white/90 font-bold uppercase tracking-widest px-8 h-12">
              <Upload className="w-4 h-4 mr-2" />
              Upload Video
            </Button>
          </Link>
          <Link to="/webcam">
            <Button className="bg-[#44FF44] text-black hover:bg-[#33CC33] font-bold uppercase tracking-widest px-8 h-12">
              <Camera className="w-4 h-4 mr-2" />
              Live Webcam
            </Button>
          </Link>
          <Link to="/dashboard">
            <Button className="bg-white text-black hover:bg-white/90 font-bold uppercase tracking-widest px-8 h-12">
              <Activity className="w-4 h-4 mr-2" />
              Live Dashboard
            </Button>
          </Link>
        </motion.div>
      </div>

      {/* Features Grid */}
      <div className="max-w-6xl w-full grid grid-cols-1 md:grid-cols-3 gap-6">
        <FeatureCard 
          icon={Target}
          title="Precision Detection"
          description="Powered by COCO-SSD neural architecture for high-accuracy object classification across 80+ categories."
        />
        <FeatureCard 
          icon={Activity}
          title="Persistent Tracking"
          description="Advanced centroid tracking algorithms maintain object identity even through brief occlusions or motion blur."
        />
        <FeatureCard 
          icon={Shield}
          title="Secure Processing"
          description="All neural inference is performed locally in-browser. Your data never leaves your secure environment."
        />
      </div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description }: { icon: any, title: string, description: string }) {
  return (
    <Card className="hardware-card border-none p-8 space-y-4">
      <div className="w-12 h-12 rounded-lg bg-white/5 flex items-center justify-center">
        <Icon className="w-6 h-6 text-[#44FF44]" />
      </div>
      <h3 className="text-lg font-bold uppercase tracking-tight text-white">{title}</h3>
      <p className="text-white/30 text-xs leading-relaxed">{description}</p>
    </Card>
  );
}
