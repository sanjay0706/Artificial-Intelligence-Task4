import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Camera, Home, Upload, BarChart3, Shield } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/webcam', label: 'Live Webcam', icon: Camera },
    { path: '/results', label: 'Results', icon: BarChart3 },
    { path: '/dashboard', label: 'Dashboard', icon: Shield },
  ];

  return (
    <nav className="w-full bg-slate-900 border-b border-white/10 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <Shield className="w-6 h-6 text-[#44FF44]" />
          <span className="text-xl font-bold tracking-tighter uppercase italic text-white">VisionTrack AI</span>
        </Link>
        
        <div className="flex gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all text-xs font-bold uppercase tracking-widest ${
                  isActive 
                    ? 'bg-[#44FF44] text-black' 
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden md:inline">{item.label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
