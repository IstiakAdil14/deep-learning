'use client';

import { ShieldAlert, ShieldCheck, Shield } from 'lucide-react';

interface RiskCardProps {
  riskLevel: string;
  predictionTime: string;
}

export function RiskCard({ riskLevel, predictionTime }: RiskCardProps) {
  const getRiskStyles = () => {
    switch (riskLevel) {
      case 'Low': 
        return {
          bg: 'bg-emerald-500/10',
          border: 'border-emerald-500/30',
          text: 'text-emerald-400',
          glow: 'shadow-[0_0_15px_rgba(16,185,129,0.1)]',
          icon: <ShieldCheck className="w-8 h-8 text-emerald-400" />
        };
      case 'Medium': 
        return {
          bg: 'bg-amber-500/10',
          border: 'border-amber-500/30',
          text: 'text-amber-400',
          glow: 'shadow-[0_0_15px_rgba(245,158,11,0.1)]',
          icon: <Shield className="w-8 h-8 text-amber-400" />
        };
      case 'High': 
        return {
          bg: 'bg-rose-500/10',
          border: 'border-rose-500/30',
          text: 'text-rose-400',
          glow: 'shadow-[0_0_20px_rgba(225,29,72,0.2)] animate-pulse-slow',
          icon: <ShieldAlert className="w-8 h-8 text-rose-400 animate-bounce" />
        };
      default: 
        return {
          bg: 'bg-slate-500/10',
          border: 'border-slate-500/30',
          text: 'text-slate-400',
          glow: '',
          icon: <Shield className="w-8 h-8 text-slate-400" />
        };
    }
  };

  const styles = getRiskStyles();

  return (
    <div className={`glass-panel p-4 flex flex-col items-center justify-center text-center h-full border ${styles.border} ${styles.glow} transition-all duration-500`}>
      <h2 className="text-lg font-semibold text-slate-400 uppercase tracking-widest mb-1">Grid Status</h2>
      <p className="text-[15px] text-slate-500 mb-3 font-mono">T+4H: {predictionTime}</p>
      
      <div className={`mb-3 p-3 rounded-full ${styles.bg}`}>
        {styles.icon}
      </div>
      
      <div className="mt-1">
        <span className={`text-4xl font-black tracking-tight ${styles.text}`}>
          {riskLevel}
        </span>
      </div>
    </div>
  );
}