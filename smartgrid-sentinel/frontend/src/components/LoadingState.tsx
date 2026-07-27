'use client';

import { Activity } from 'lucide-react';

export function LoadingState() {
  return (
    <div className="flex flex-col justify-center items-center py-20 animate-pulse-slow">
      <div className="relative w-24 h-24 mb-6">
        {/* Outer glowing ring */}
        <div className="absolute inset-0 border-4 border-slate-600 rounded-full"></div>
        {/* Spinning gradient ring */}
        <div className="absolute inset-0 border-4 border-transparent border-t-blue-500 border-r-teal-400 rounded-full animate-spin shadow-[0_0_15px_rgba(59,130,246,0.3)]"></div>
        {/* Inner icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <Activity className="w-8 h-8 text-blue-400 animate-pulse" />
        </div>
      </div>
      
      <h3 className="text-xl font-semibold text-slate-200 tracking-wider">Processing Telemetry</h3>
      <p className="text-sm text-slate-400 mt-2 font-mono">Running neural network inference...</p>
    </div>
  );
}