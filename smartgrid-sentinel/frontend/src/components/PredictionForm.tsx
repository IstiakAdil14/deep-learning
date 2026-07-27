'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { MapPin, Search, Cpu } from 'lucide-react';

interface PredictionFormProps {
  onPredict: (district: string, upazila: string) => void;
}

export function PredictionForm({ onPredict }: PredictionFormProps) {
  const [districts, setDistricts] = useState<string[]>([]);
  const [upazilas, setUpazilas] = useState<string[]>([]);
  const [selectedDistrict, setSelectedDistrict] = useState('');
  const [selectedUpazila, setSelectedUpazila] = useState('');

  useEffect(() => {
    fetchDistricts();
  }, []);

  const fetchDistricts = async () => {
    try {
      const response = await axios.get('http://localhost:8000/districts');
      setDistricts(response.data);
    } catch (error) {
      console.error('Error fetching districts:', error);
      setDistricts(['Sylhet', 'Habiganj', 'Moulvibazar', 'Sunamganj']);
    }
  };

  const fetchUpazilas = async (district: string) => {
    try {
      const response = await axios.get(`http://localhost:8000/upazilas/${district}`);
      setUpazilas(response.data);
    } catch (error) {
      console.error('Error fetching upazilas:', error);
      const fallbackUpazilas: { [key: string]: string[] } = {
        'Sylhet': ['Beanibazar', 'Sylhet Sadar', 'Jaintiapur'],
        'Habiganj': ['Madhabpur', 'Habiganj Sadar', 'Ajmiriganj'],
        'Moulvibazar': ['Sreemangal', 'Moulvibazar Sadar'],
        'Sunamganj': ['Tahirpur', 'Sunamganj Sadar', 'Chhatak']
      };
      setUpazilas(fallbackUpazilas[district] || []);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedDistrict && selectedUpazila) {
      onPredict(selectedDistrict, selectedUpazila);
    }
  };

  return (
    <div className="glass-panel p-5 relative overflow-hidden group">
      {/* Decorative background element */}
      <div className="absolute -right-10 -top-10 w-24 h-24 bg-blue-500/10 rounded-full blur-2xl group-hover:bg-blue-400/20 transition-all duration-700"></div>
      
      <div className="flex items-center mb-4 border-b border-slate-600/50 pb-3">
        <Cpu className="text-blue-400 w-5 h-5 mr-2" />
        <h2 className="text-lg font-medium text-slate-200 tracking-wide">Target Asset Location</h2>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 relative z-10">
          <div>
            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
              District Region
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <MapPin className="h-4 w-4 text-slate-500" />
              </div>
              <select
                value={selectedDistrict}
                onChange={(e) => {
                  setSelectedDistrict(e.target.value);
                  setSelectedUpazila('');
                  fetchUpazilas(e.target.value);
                }}
                className="glass-input block w-full pl-9 pr-3 py-2 text-sm"
                required
              >
                <option value="" className="bg-slate-800 text-slate-400">Select District Grid</option>
                {districts.map((d) => (
                  <option key={d} value={d} className="bg-slate-800">{d}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
              Upazila Sub-station
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <MapPin className="h-4 w-4 text-slate-500" />
              </div>
              <select
                value={selectedUpazila}
                onChange={(e) => setSelectedUpazila(e.target.value)}
                className="glass-input block w-full pl-9 pr-3 py-2 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                required
                disabled={!selectedDistrict}
              >
                <option value="" className="bg-slate-800 text-slate-400">Select Sub-station</option>
                {upazilas.map((u) => (
                  <option key={u} value={u} className="bg-slate-800">{u}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <button
          type="submit"
          disabled={!selectedDistrict || !selectedUpazila}
          className="mt-5 w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-sm font-medium py-2.5 px-4 rounded-lg hover:from-blue-500 hover:to-indigo-500 transition-all duration-300 flex items-center justify-center shadow-md shadow-blue-900/40 disabled:opacity-50 disabled:cursor-not-allowed transform hover:-translate-y-0.5"
        >
          <Search className="w-4 h-4 mr-2" />
          Execute Telemetry Analysis
        </button>
      </form>
    </div>
  );
}