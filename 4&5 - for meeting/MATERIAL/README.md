# ⚡ SmartGrid Sentinel
## Predictive Load Shedding Risk Forecasting & Intelligent Alert System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-UI-38BDF8)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Development-yellow)

---

# 📌 Project Overview

SmartGrid Sentinel is a hybrid AI-powered energy risk forecasting platform designed to predict potential load shedding risks across Bangladesh at the Upazila level.

The project combines:

- Deep Learning (LSTM)
- NLP-based alert explanation
- Event-driven alert mechanisms
- Real historical weather data
- Smart-grid inspired forecasting architecture

The platform predicts electricity instability risks using time-series learning and provides intelligent human-readable alerts through a modern web dashboard.

---

# 🎯 Project Objectives

## Primary Objectives

- Forecast load shedding risk using Deep Learning
- Provide Upazila-level predictive monitoring
- Generate intelligent alerts and warnings
- Build a scalable hybrid AI architecture

## Secondary Objectives

- Improve AI interpretability
- Demonstrate smart-grid inspired workflows
- Simulate realistic electricity demand behavior
- Support localized energy awareness systems

---

# 🧠 Core Technologies

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Forecasting Model | LSTM |
| NLP Layer | Rule-Based NLG |
| Data Processing | Pandas, NumPy |
| Geospatial Processing | GeoPandas |
| Frontend | Next.js + TypeScript |
| UI Framework | Tailwind CSS |
| Animations | Framer Motion |
| Backend API | FastAPI / Node.js |
| Visualization | Chart.js / Recharts |

---

# 🏗️ System Architecture

```text
Administrative Data + Real Weather Data
                    ↓
         Demand Simulation Engine
                    ↓
           Dataset Preparation
                    ↓
           LSTM Forecasting Model
                    ↓
          Risk Prediction Output
                    ↓
        Event-Driven Alert Engine
                    ↓
      NLP Explanation Generation
                    ↓
            Web Dashboard
```

---

# 🌍 Dataset Information

## Dataset Characteristics

| Property | Details |
|---|---|
| Time Range | Apr 13 → May 13, 2026 |
| Interval | 2-hour |
| Upazilas | 477 |
| Districts | 64 |
| Divisions | 8 |
| Total Rows | 181,536 |
| Weather Data | Real |
| Electricity Demand | Simulated |
| Null Values | 0 |

---

# 📊 Dataset Features

| Feature | Description |
|---|---|
| datetime | Timestamp |
| division | Division name |
| district | District name |
| upazila | Upazila name |
| temperature | Real weather temperature |
| humidity | Real humidity |
| rainfall | Real rainfall |
| demand_index | Simulated electricity demand |
| risk_level | Low / Medium / High |

---

# 🌤️ Weather Data Source

Real historical weather data is collected from:

- Open-Meteo Historical API

Weather features:

- Temperature
- Humidity
- Rainfall

District-level weather is propagated to corresponding Upazilas for scalable localized modeling.

---

# ⚡ Demand Simulation Strategy

Since real Upazila-level electricity datasets are not publicly available in Bangladesh, electricity demand behavior is synthetically modeled using:

- Time-of-day patterns
- Weather influence
- Peak-hour demand logic
- Controlled stochastic noise

The system simulates realistic smart-grid demand behavior for forecasting research purposes.

---

# 🤖 Deep Learning Model

## Model Used

### Long Short-Term Memory (LSTM)

LSTM is used because:

- Electricity demand is sequential
- Temporal dependency exists in usage patterns
- Weather impacts vary over time
- Historical behavior influences future demand

---

# 🧠 NLP Explanation Engine

The NLP layer converts prediction outputs into human-readable alert messages.

### Example Output

> High load shedding risk expected in Kaliakair Upazila due to elevated evening demand and increased temperature.

---

# 🚨 Event-Driven Alert System

The alert engine continuously monitors prediction outputs and automatically generates warnings based on predefined thresholds.

| Risk Level | Alert Type |
|---|---|
| High | Critical Alert |
| Medium | Warning Alert |
| Low | Stable Status |

---

# 👥 User Types

| User Type | Responsibilities |
|---|---|
| General Users | View forecasts and alerts |
| Business Users | Monitor operational electricity risks |
| Admin | Manage monitoring and analytics |

---

# 💻 Frontend Features

## Planned Features

- Modern responsive dashboard
- Real-time risk cards
- Upazila search and filtering
- Interactive analytics charts
- Alert feed system
- Dark/light UI support
- Mobile responsive design

---

# 📁 Project Structure

```text
smartgrid-sentinel/
│
├── backend/
│   ├── api/
│   ├── models/
│   ├── dataset/
│   ├── forecasting/
│   └── nlp/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── lib/
│   └── styles/
│
├── notebooks/
│   ├── dataset_generation.ipynb
│   ├── lstm_training.ipynb
│   └── evaluation.ipynb
│
├── datasets/
│   └── smartgrid_sentinel_dataset.csv
│
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/IstiakAdil14/smartgrid-sentinel.git
```

---

## Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

# 🚀 Future Enhancements

- Real-time weather streaming
- Push notification system
- GIS heatmap integration
- Smart meter integration
- Transformer-based forecasting
- Reinforcement learning optimization
- Mobile application support

---

# 🧪 Research Contribution

This project demonstrates:

- Hybrid AI system integration
- Localized smart-grid forecasting
- Explainable AI alert generation
- Weather-aware energy prediction
- Bangladesh-focused intelligent infrastructure modeling

---

# 📚 Academic Context

This project was developed as part of:

**Course:** Deep Learning (CSE-460)

**Department:** Computer Science and Engineering  
**Institution:** North East University Bangladesh

---

# 👨‍💻 Author

## Md. Istiak Hussain Adil

- ID: 0562220005101053
- Department of Computer Science and Engineering
- North East University Bangladesh

---

# 📄 License

This project is developed for academic and research purposes.

---

# ⭐ Final Note

SmartGrid Sentinel is designed as a modern AI-driven smart-grid forecasting platform combining Deep Learning, NLP, and intelligent alert generation to simulate localized electricity risk monitoring in Bangladesh.

---

# 📚 References

[1] Z. C. Lipton, J. Berkowitz, and C. Elkan, "A Critical Review of Recurrent Neural Networks for Sequence Learning," *arXiv preprint arXiv:1506.00019v4*, Oct. 2015. [Online]. Available: https://arxiv.org/abs/1506.00019

> Foundational survey on RNNs and LSTM architectures. Provides theoretical basis for the LSTM-based time-series forecasting model used in SmartGrid Sentinel for sequential electricity demand prediction.

[2] Y. Chen, Z. Tang, X. Weng, M. He, G. Zhang, D. Yuan, and T. Jin, "A Novel Approach for Evaluating Power Quality in Distributed Power Distribution Networks Using AHP and S-Transform," *Energies*, vol. 17, no. 2, p. 411, Jan. 2024. https://doi.org/10.3390/en17020411

> Presents a comprehensive power quality assessment framework for distributed energy networks. Informs the risk classification and alert threshold design in SmartGrid Sentinel's event-driven alert engine.

[3] M. Darvishi, M. Tahmasebi, E. Shokouhmand, J. Pasupuleti, P. Bokoro, and J. S. Raafat, "Optimal Operation of Sustainable Virtual Power Plant Considering the Amount of Emission in the Presence of Renewable Energy Sources and Demand Response," *Sustainability*, vol. 15, no. 14, p. 11012, Jul. 2023. https://doi.org/10.3390/su151411012

> Examines smart grid demand management under renewable energy uncertainty using optimization and demand response programs. Supports the demand simulation strategy and peak-hour load modeling methodology adopted in SmartGrid Sentinel.
