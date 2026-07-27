"""
Grid Assets Module
Version 2 - Added Upazila support

Generates location-based grid asset features.
"""

import random

from config import SEED
from locations import DIVISION
from locations import LOCATIONS

random.seed(SEED)


# --------------------------------------------------
# Location Selection
# --------------------------------------------------

def get_random_location():
    """Select a random district and upazila."""
    district = random.choice(list(LOCATIONS.keys()))
    upazila = random.choice(LOCATIONS[district])
    return district, upazila


# Urban/Rural classification based on upazila
URBAN_UPAZILAS = {
    "Sylhet": ["Sylhet Sadar", "Osmani Nagar"],
    "Moulvibazar": ["Moulvibazar Sadar"],
    "Habiganj": ["Habiganj Sadar", "Ajmiriganj"],
    "Sunamganj": ["Sunamganj Sadar"]
}


def get_area_type(district, upazila):
    """Determine if area is Urban or Rural."""
    if district in URBAN_UPAZILAS:
        if upazila in URBAN_UPAZILAS[district]:
            return "Urban"
    return "Rural"


# --------------------------------------------------
# Substation ID
# --------------------------------------------------

def get_substation_id():
    """Generate a random substation ID."""
    ss_num = random.randint(1, 50)
    return f"SS_{ss_num:03d}"


# --------------------------------------------------
# Feeder ID
# --------------------------------------------------

def get_feeder_id():
    """Generate a random feeder ID."""
    fdr_num = random.randint(1, 20)
    return f"FDR_{fdr_num:02d}"


# --------------------------------------------------
# Transformer Age
# --------------------------------------------------

def get_transformer_age():
    """Generate transformer age in years (5-25 years)."""
    return random.randint(5, 25)


# --------------------------------------------------
# Transformer Capacity
# --------------------------------------------------

def get_transformer_capacity():
    """Generate transformer capacity in MVA."""
    # Standard transformer capacities
    capacities = [200, 250, 300, 350, 400]
    return random.choice(capacities)


# --------------------------------------------------
# Outage History
# --------------------------------------------------

def get_outage_history():
    """Generate count of past outages in last year (0-10)."""
    return random.randint(0, 10)


# --------------------------------------------------
# Maintenance Due
# --------------------------------------------------

def get_maintenance_due(transformer_age, outage_history):
    """Determine if maintenance is due."""
    # Older transformers or those with frequent outages need maintenance
    maintenance_prob = 0.2 + (transformer_age - 5) * 0.03 + outage_history * 0.02
    return "Yes" if random.random() < maintenance_prob else "No"


# --------------------------------------------------
# Population Density
# --------------------------------------------------

def get_population_density(area_type):
    """Generate population density (people per sq km)."""
    if area_type == "Urban":
        return random.randint(5000, 20000)
    else:
        return random.randint(500, 2000)


# --------------------------------------------------
# Industrial Load Ratio
# --------------------------------------------------

def get_industrial_load_ratio(area_type):
    """Generate ratio of industrial load (0-0.6)."""
    if area_type == "Urban":
        return round(random.uniform(0.4, 0.6), 2)
    else:
        return round(random.uniform(0.0, 0.3), 2)


# --------------------------------------------------
# Main Grid Features
# --------------------------------------------------

def get_grid_features():
    """Generate all grid asset features."""
    district, upazila = get_random_location()
    area_type = get_area_type(district, upazila)
    transformer_age = get_transformer_age()
    outage_history = get_outage_history()
    
    return {
        "district": district,
        "upazila": upazila,
        "area_type": area_type,
        "substation_id": get_substation_id(),
        "feeder_id": get_feeder_id(),
        "transformer_age": transformer_age,
        "transformer_capacity": get_transformer_capacity(),
        "outage_history": outage_history,
        "maintenance_due": get_maintenance_due(transformer_age, outage_history),
        "population_density": get_population_density(area_type),
        "industrial_load_ratio": get_industrial_load_ratio(area_type)
    }