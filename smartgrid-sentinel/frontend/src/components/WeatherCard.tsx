'use client';

import { Thermometer, Droplet, CloudRain, Wind, Sun, Cloud, CloudLightning } from 'lucide-react';

interface WeatherCardProps {
  weather: {
    temperature: number;
    humidity: number;
    rainfall: number;
    wind_speed: number;
    condition: string;
  };
}

export function WeatherCard({ weather }: WeatherCardProps) {
  const getWeatherIcon = () => {
    switch(weather.condition) {
      case 'Sunny': return <Sun className="w-8 h-8 text-yellow-400" />;
      case 'Cloudy': return <Cloud className="w-8 h-8 text-slate-400" />;
      case 'Rainy': return <CloudRain className="w-8 h-8 text-blue-400" />;
      case 'Stormy': return <CloudLightning className="w-8 h-8 text-indigo-400" />;
      default: return <Sun className="w-8 h-8 text-yellow-400" />;
    }
  };

  return (
    <div className="glass-panel p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4 border-b border-slate-600/50 pb-3">
        <div>
          <h2 className="text-base font-medium text-slate-200">Environmental Data</h2>
          <p className="text-[10px] text-slate-500 font-mono mt-0.5">LIVE METRICS</p>
        </div>
        <div className="p-2 bg-slate-800/80 rounded-lg border border-slate-600/50 shadow-sm">
          {getWeatherIcon()}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 flex-grow">
        {/* Temperature */}
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-600/30 hover:border-blue-500/40 hover:bg-slate-700/50 transition-all duration-300 group shadow-sm">
          <div className="flex items-center mb-1">
            <Thermometer className="w-3.5 h-3.5 text-rose-400 mr-1.5 group-hover:scale-110 transition-transform" />
            <span className="text-[20px] font-semibold text-slate-400 uppercase">Temp</span>
          </div>
          <div className="flex items-baseline">
            <span className="text-5xl font-bold text-slate-200">{weather.temperature}</span>
            <span className="text-xs text-slate-500 ml-1">°C</span>
          </div>
        </div>

        {/* Humidity */}
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-600/30 hover:border-blue-500/40 hover:bg-slate-700/50 transition-all duration-300 group shadow-sm">
          <div className="flex items-center mb-1">
            <Droplet className="w-3.5 h-3.5 text-blue-400 mr-1.5 group-hover:scale-110 transition-transform" />
            <span className="text-[20px] font-semibold text-slate-400 uppercase">Humidity</span>
          </div>
          <div className="flex items-baseline">
            <span className="text-5xl font-bold text-slate-200">{weather.humidity}</span>
            <span className="text-xs text-slate-500 ml-1">%</span>
          </div>
        </div>

        {/* Rainfall */}
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-600/30 hover:border-blue-500/40 hover:bg-slate-700/50 transition-all duration-300 group shadow-sm">
          <div className="flex items-center mb-1">
            <CloudRain className="w-3.5 h-3.5 text-indigo-400 mr-1.5 group-hover:scale-110 transition-transform" />
            <span className="text-[20px] font-semibold text-slate-400 uppercase">Rainfall</span>
          </div>
          <div className="flex items-baseline">
            <span className="text-5xl font-bold text-slate-200">{weather.rainfall}</span>
            <span className="text-xs text-slate-500 ml-1">mm</span>
          </div>
        </div>

        {/* Wind Speed */}
        <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-600/30 hover:border-blue-500/40 hover:bg-slate-700/50 transition-all duration-300 group shadow-sm">
          <div className="flex items-center mb-1">
            <Wind className="w-3.5 h-3.5 text-teal-400 mr-1.5 group-hover:scale-110 transition-transform" />
            <span className="text-[20px] font-semibold text-slate-400 uppercase">Wind</span>
          </div>
          <div className="flex items-baseline">
            <span className="text-5xl font-bold text-slate-200">{weather.wind_speed}</span>
            <span className="text-xs text-slate-500 ml-1">m/s</span>
          </div>
        </div>
      </div>
    </div>
  );
}