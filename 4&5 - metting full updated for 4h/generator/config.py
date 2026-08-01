"""
Global configuration
"""

from datetime import datetime

# ===========================================
# Dataset Period
# ===========================================

START_DATE = datetime(2026, 4, 13, 0, 0)
END_DATE   = datetime(2026, 5, 12, 22, 0)

INTERVAL_HOURS = 2

# ===========================================
# Dataset Information
# ===========================================

DIVISION = "Sylhet"

# ===========================================
# Weather Limits
# ===========================================

TEMP_MIN = 18
TEMP_MAX = 36

HUMIDITY_MIN = 45
HUMIDITY_MAX = 100

RAIN_MAX = 120

WIND_MAX = 60

# ===========================================
# Grid Limits
# ===========================================

DEMAND_MIN = 100
DEMAND_MAX = 280

CAPACITY_MIN = 130
CAPACITY_MAX = 300

TRANSFORMER_MAX = 110

GRID_MAX = 100

# ===========================================
# Random Seed
# ===========================================

SEED = 42