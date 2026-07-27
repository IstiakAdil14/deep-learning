'use client';

import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell, CartesianGrid } from 'recharts';

interface ProbabilityChartProps {
  confidence: {
    Low: number;
    Medium: number;
    High: number;
  };
}

export function ProbabilityChart({ confidence }: ProbabilityChartProps) {
  const data = [
    { name: 'Low', value: confidence.Low * 100, color: '#34d399' }, // Emerald-400
    { name: 'Medium', value: confidence.Medium * 100, color: '#fbbf24' }, // Amber-400
    { name: 'High', value: confidence.High * 100, color: '#fb7185' }, // Rose-400
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800/90 backdrop-blur-sm border border-slate-600/50 p-2.5 rounded-md shadow-lg shadow-slate-900/50">
          <p className="text-slate-200 text-sm font-semibold">{`${payload[0].payload.name} Risk`}</p>
          <p className="text-blue-400 text-xs font-mono mt-0.5">{`Probability: ${payload[0].value.toFixed(1)}%`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-panel p-4 h-full flex flex-col">
      <h2 className="text-lg font-semibold text-slate-400 uppercase tracking-widest mb-3">Confidence Distribution</h2>
      
      <div className="flex-grow w-full min-h-[160px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#475569" opacity={0.5} />
            <XAxis 
              type="number" 
              domain={[0, 100]} 
              tickFormatter={(v) => `${v}%`} 
              stroke="#94a3b8" 
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              type="category" 
              dataKey="name" 
              stroke="#cbd5e1" 
              fontSize={11}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(71, 85, 105, 0.3)'}} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16} animationDuration={1500}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}