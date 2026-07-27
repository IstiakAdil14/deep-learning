'use client';

import { AlertTriangle, CheckCircle2, ChevronRight, Zap } from 'lucide-react';

interface RecommendationCardProps {
  recommendations: string[];
}

export function RecommendationCard({ recommendations }: RecommendationCardProps) {
  const isHighRisk = recommendations.some(r => r.includes('⚠'));

  return (
    <div className="glass-panel p-4 h-full flex flex-col">
      <div className="flex items-center mb-4 pb-3 border-b border-slate-600/50">
        <div className={`p-1.5 rounded-lg mr-2 shadow-sm ${isHighRisk ? 'bg-rose-500/20 text-rose-400' : 'bg-blue-500/20 text-blue-400'}`}>
          {isHighRisk ? <AlertTriangle className="w-6 h-6" /> : <Zap className="w-6 h-6" />}
        </div>
        <h2 className="text-xl font-medium text-slate-200">Action Protocols</h2>
      </div>
      
      <div className="flex-grow flex flex-col justify-center">
        <ul className="space-y-3">
          {recommendations.map((rec, index) => {
            const isWarning = rec.includes('⚠');
            const cleanText = rec.replace('⚠ ', '').replace('✓ ', '');
            
            return (
              <li 
                key={index} 
                className={`flex items-start p-2 rounded-lg border transition-all duration-300 shadow-sm ${
                  isWarning 
                    ? 'bg-rose-500/10 border-rose-500/30 hover:bg-rose-500/20' 
                    : 'bg-slate-800/50 border-slate-600/30 hover:bg-slate-700/60'
                }`}
              >
                {isWarning ? (
                  <AlertTriangle className="w-5 h-5 text-rose-400 mt-0.5 mr-2 flex-shrink-0" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5 mr-2 flex-shrink-0" />
                )}
                
                <div>
                  <p className={`text-lg ${isWarning ? 'text-rose-200 font-medium' : 'text-slate-300'}`}>
                    {cleanText}
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
      
      {isHighRisk && (
        <div className="mt-4 p-2 bg-rose-500/10 border border-rose-500/30 rounded-lg flex items-center justify-between group cursor-pointer hover:bg-rose-500/20 transition-all shadow-sm">
          <span className="text-sm font-semibold text-rose-400 uppercase tracking-wider">Execute Emergency Protocol</span>
          <ChevronRight className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" />
        </div>
      )}
    </div>
  );
}