/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import UploadPage from './pages/Upload';
import WebcamPage from './pages/Webcam';
import ResultsPage from './pages/Results';
import VisionDashboard from './components/VisionDashboard';

export default function App() {
  return (
    <Router>
      <div className="dark min-h-screen bg-[#0a0a0b] text-white font-sans selection:bg-[#44FF44] selection:text-black">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/webcam" element={<WebcamPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/dashboard" element={<VisionDashboard />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

