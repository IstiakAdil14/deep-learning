"use client";

import { useState } from 'react';
import axios from 'axios';
import { PredictionForm } from '@/components/PredictionForm';
import { WeatherCard } from '@/components/WeatherCard';
import { RiskCard } from '@/components/RiskCard';
import { ProbabilityChart } from '@/components/ProbabilityChart';
import { RecommendationCard } from '@/components/RecommendationCard';
import { LoadingState } from '@/components/LoadingState';

interface PredictionResult {
  risk_level: string;
  confidence: {
    Low: number;
    Medium: number;
    High: number;
  };
  weather: {
    temperature: number;
    humidity: number;
    rainfall: number;
    wind_speed: number;
    condition: string;
  };
  prediction_time: string;
  recommendation: string[];
}

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (district: string, upazila: string) => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        district,
        upazila
      });
      setPrediction(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-12">
      <div className="ambient-glow-1"></div>
      <div className="ambient-glow-2"></div>
      
      <div className="max-w-6xl mx-auto px-4 relative z-10">
        {/* Hero Section */}
        <div className="text-center mb-12 animate-float">
          <div className="inline-block mb-4 p-2 rounded-full bg-blue-500/10 border border-blue-500/20">
            <span className="text-blue-400 font-semibold text-sm tracking-widest uppercase px-4">AI Powered</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-teal-400 to-indigo-400 mb-4 drop-shadow-sm">
            SmartGrid Sentinel
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto font-light">
            Advanced neural network telemetry predicting load shedding and grid instability for the next 4 hours.
          </p>
        </div>

        {/* Prediction Form */}
        <div className="max-w-3xl mx-auto mb-12">
          <PredictionForm onPredict={handlePredict} />
        </div>

        {/* Loading State */}
        {loading && <LoadingState />}

        {/* Results Dashboard */}
        {prediction && !loading && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8 animate-fade-in">
            {/* Left Column (Weather) */}
            <div className="lg:col-span-1">
              <WeatherCard weather={prediction.weather} />
            </div>

            {/* Middle Column (Risk & Chart) */}
            <div className="lg:col-span-1 flex flex-col gap-6">
              <RiskCard 
                riskLevel={prediction.risk_level} 
                predictionTime={prediction.prediction_time}
              />
              <ProbabilityChart confidence={prediction.confidence} />
            </div>

            {/* Right Column (Recommendations) */}
            <div className="lg:col-span-1">
              <RecommendationCard recommendations={prediction.recommendation} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}