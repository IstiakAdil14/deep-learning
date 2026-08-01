#!/usr/bin/env python3
"""
SmartGrid Sentinel - Explainable AI & Recommendation Module (NLP / NLG)

This module is an Explainable AI (XAI) layer placed ON TOP of the existing
supervised ML / Deep Learning prediction models (Random Forest, XGBoost,
Bidirectional LSTM). It converts a numerical prediction (risk level + confidence)
into human-readable Natural Language Generation (NLG) output:

    1. AI Explanation      - why the model predicts this risk
    2. AI Recommendation   - what operators should do
    3. Natural Language Query parsing - extract location / time / intent
    4. AI Report Generator - full printable SmartGrid Sentinel report

No synthetic text is added to the training dataset and no model is retrained.
The prediction models remain untouched.
"""

import re


# ---------------------------------------------------------------------------
# Knowledge thresholds (derived from domain reasoning on the smart-grid data)
# ---------------------------------------------------------------------------
# These thresholds are used to explain predictions in plain language. They are
# not used to train or alter the prediction model in any way.
DEMAND_CAPACITY_WARN = 1.0      # demand >= capacity  -> deficit
DEMAND_CAPACITY_SEVERE = 1.2    # demand > 1.2 x capacity -> critical deficit
TRANSFORMER_WARN = 80.0
TRANSFORMER_SEVERE = 95.0
GRID_STABILITY_WARN = 85.0
GRID_STABILITY_SEVERE = 50.0
OUTAGE_WARN = 2


# ---------------------------------------------------------------------------
# 1. AI EXPLANATION (Natural Language Generation)
# ---------------------------------------------------------------------------
def generate_explanation(data):
    """Return a list of human-readable bullet reasons for the predicted risk.

    `data` contains the raw grid parameters used by the prediction pipeline.
    """
    reasons = []

    demand = data.get('electricity_demand', 0.0)
    capacity = max(data.get('generation_capacity', 0.0), 1.0)
    ratio = demand / capacity

    if ratio >= DEMAND_CAPACITY_SEVERE:
        deficit = demand - capacity
        reasons.append(
            f"Electricity demand exceeds generation capacity by "
            f"{deficit:.0f} MW (ratio {ratio:.2f}x), indicating a critical supply deficit."
        )
    elif ratio >= DEMAND_CAPACITY_WARN:
        deficit = demand - capacity
        reasons.append(
            f"Electricity demand meets or exceeds generation capacity "
            f"({deficit:+.0f} MW, ratio {ratio:.2f}x), leaving little reserve margin."
        )
    else:
        reasons.append(
            f"Electricity demand is within generation capacity "
            f"(ratio {ratio:.2f}x), so supply margin is currently adequate."
        )

    tload = data.get('transformer_load', 0.0)
    if tload >= TRANSFORMER_SEVERE:
        reasons.append(
            f"Transformer loading is critically high at {tload:.0f}% "
            f"(safe limit ~80%), risking thermal overload."
        )
    elif tload >= TRANSFORMER_WARN:
        reasons.append(
            f"Transformer loading is elevated at {tload:.0f}% "
            f"(above the {TRANSFORMER_WARN:.0f}% comfort threshold)."
        )
    else:
        reasons.append(
            f"Transformer loading is moderate at {tload:.0f}%, "
            f"within safe operating limits."
        )

    stability = data.get('grid_stability', 0.0)
    if stability < GRID_STABILITY_SEVERE:
        reasons.append(
            f"Grid stability score has fallen to {stability:.0f} "
            f"(below the {GRID_STABILITY_SEVERE:.0f} critical floor), "
            f"signalling a fragile network."
        )
    elif stability < GRID_STABILITY_WARN:
        reasons.append(
            f"Grid stability score is low at {stability:.0f} "
            f"(below the {GRID_STABILITY_WARN:.0f} healthy threshold)."
        )
    else:
        reasons.append(
            f"Grid stability score is healthy at {stability:.0f}."
        )

    outages = data.get('outages_24h', 0)
    if outages > OUTAGE_WARN:
        reasons.append(
            f"Recent outage frequency is high ({outages} in last 24h), "
            f"indicating already-stressed infrastructure."
        )
    elif outages > 0:
        reasons.append(
            f"{outages} outage(s) reported in the last 24h."
        )

    rainfall = data.get('rainfall', 0.0)
    wind = data.get('wind_speed', 0.0)
    weather_notes = []
    if rainfall > 20:
        weather_notes.append(f"heavy rainfall ({rainfall:.0f} mm)")
    if wind > 50:
        weather_notes.append(f"high wind speed ({wind:.0f} km/h)")
    if weather_notes:
        reasons.append(
            "Adverse weather (" + ", ".join(weather_notes) +
            ") increases the likelihood of faults and load-shedding."
        )

    return reasons


# ---------------------------------------------------------------------------
# 2. AI RECOMMENDATION (Rule-based + NLG)
# ---------------------------------------------------------------------------
def generate_recommendation(data, risk):
    """Return an ordered list of operational recommendations for `risk`."""
    recs = []

    ratio = data.get('electricity_demand', 0.0) / max(
        data.get('generation_capacity', 0.0), 1.0
    )
    tload = data.get('transformer_load', 0.0)
    stability = data.get('grid_stability', 0.0)

    if risk == "High":
        if ratio >= DEMAND_CAPACITY_WARN:
            recs.append("Activate reserve / peaking generators immediately to close the supply gap.")
            recs.append("Request load relief from the regional grid operator.")
        if tload >= TRANSFORMER_WARN:
            recs.append("Reduce industrial and bulk consumer load to relieve transformer stress.")
            recs.append("Closely monitor transformer temperatures and deploy cooling if available.")
        if stability < GRID_STABILITY_SEVERE:
            recs.append("Island or sectionalise vulnerable feeders to protect grid stability.")
        recs.append("Notify consumers proactively of possible scheduled load-shedding.")
        recs.append("Mobilise field crews for rapid fault response.")
    elif risk == "Medium":
        if ratio >= DEMAND_CAPACITY_WARN:
            recs.append("Prepare reserve generation and keep it on hot standby.")
        if tload >= TRANSFORMER_WARN:
            recs.append("Shed non-critical / discretionary load where possible.")
            recs.append("Increase transformer monitoring frequency.")
        if stability < GRID_STABILITY_WARN:
            recs.append("Review stability margins and pre-position compensation equipment.")
        recs.append("Issue a caution advisory to consumers in the affected area.")
    else:  # Low
        recs.append("Maintain routine monitoring; no immediate action required.")
        if tload >= TRANSFORMER_WARN or stability < GRID_STABILITY_WARN:
            recs.append("Schedule preventive maintenance during the next low-demand window.")
        recs.append("Continue standard grid surveillance.")

    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


# ---------------------------------------------------------------------------
# 3. NATURAL LANGUAGE QUERY PARSING
# ---------------------------------------------------------------------------
# Known locations from the dataset (division is fixed to Sylhet).
KNOWN_DISTRICTS = {
    "sylhet", "habiganj", "moulvibazar", "moulvi bazar", "sunamganj", "sunam ganj"
}
KNOWN_UPAZILAS = {
    "golapganj", "golap ganj", "golapgonj", "beanibazar", "beanibazar",
    "sreemangal", "sreemongol", "zakiganj", "kanaighat", "companiganj",
    "sylhet sadar", "sylhet", "bisheshwarganj", "chhatak", "dakin surma",
    "fenchuganj", "gowainghat", "jaintiapur", "khalilabad", "madhabpur",
    "rustampur", "barlekha", "kamalganj", "kulaura", "sullah", "derai",
    "dharampasha", "jamalganj", "tahirpur", "bishwamvarpur", "charmohar"
}

TIME_PATTERN = re.compile(
    r"(?P<hour>\d{1,2})\s*(?:pm|am)|"
    r"(?P<word>morning|afternoon|evening|night|noon|midnight|tonight|today)"
)


def _word_to_hour(word):
    word = word.lower()
    table = {
        "morning": 8, "noon": 12, "afternoon": 15, "evening": 18,
        "night": 21, "midnight": 0, "tonight": 21, "today": None,
    }
    return table.get(word)


def parse_natural_language_query(text):
    """Extract (location, hour, intent) from a free-text user query.

    Returns a dict:
        {
            'location': str or None,
            'hour': int or None,
            'intent': 'predict' (default for risk-related queries)
        }
    """
    text_l = text.lower()
    result = {"location": None, "hour": None, "intent": "predict"}

    # Location extraction
    for loc in KNOWN_UPAZILAS:
        if loc in text_l:
            result["location"] = loc.title()
            break
    if result["location"] is None:
        for loc in KNOWN_DISTRICTS:
            if loc in text_l:
                result["location"] = loc.title()
                break

    # Time extraction
    m = TIME_PATTERN.search(text_l)
    if m:
        if m.group("hour"):
            h = int(m.group("hour"))
            # handle am/pm
            if "pm" in text_l and h < 12:
                h += 12
            elif "am" in text_l and h == 12:
                h = 0
            result["hour"] = h % 24
        elif m.group("word"):
            result["hour"] = _word_to_hour(m.group("word"))

    return result


# ---------------------------------------------------------------------------
# 4. AI REPORT GENERATOR
# ---------------------------------------------------------------------------
def format_hour(hour):
    """Convert an integer hour (0-23) to a 12h clock string like '7:00 PM'."""
    if hour is None:
        return "N/A"
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    suffix = "AM" if hour < 12 else "PM"
    return f"{h12}:00 {suffix}"


def format_report(data, risk, confidence):
    """Build the full SmartGrid Sentinel textual report (NLG)."""
    location = f"{data.get('upazila_name', 'N/A')}, {data.get('district_name', 'N/A')}, {data.get('division_name', 'N/A')}"

    reasons = generate_explanation(data)
    recs = generate_recommendation(data, risk)

    lines = []
    lines.append("-" * 44)
    lines.append("SMARTGRID SENTINEL REPORT")
    lines.append("-" * 44)
    lines.append("")
    lines.append(f"Location:")
    lines.append(f"  {location}")
    lines.append("")
    lines.append(f"Predicted Risk:")
    lines.append(f"  {risk}")
    lines.append("")
    lines.append(f"Confidence:")
    lines.append(f"  {confidence}")
    lines.append("")
    lines.append("Reason:")
    for r in reasons:
        lines.append(f"  - {r}")
    lines.append("")
    lines.append("Recommendation:")
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. {r}")
    lines.append("")
    lines.append("Generated by SmartGrid Sentinel AI")
    lines.append("-" * 44)

    return "\n".join(lines)


def generate_report(data, risk, confidence, parsed=None):
    """Generate the polished, viva-ready SmartGrid Sentinel AI report.

    `confidence` is a numeric value (0-100, e.g. 93.84).
    `parsed` is the optional NL-query parse dict (location/hour/intent).
    """
    # Resolve location labels (prefer NL parse if it populated a location)
    division = data.get('division_name', 'Sylhet')
    district = data.get('district_name', 'N/A')
    upazila = data.get('upazila_name', 'N/A')
    if parsed and parsed.get('location'):
        # The parser may return an upazila or district name; show it clearly.
        loc = parsed['location']
        # If it looks like a known district, treat as district; else upazila.
        if loc.lower() in KNOWN_DISTRICTS:
            district = loc
        else:
            upazila = loc

    hour = data.get('current_hour')
    if parsed and parsed.get('hour') is not None:
        hour = parsed['hour']

    reasons = generate_explanation(data)
    recs = generate_recommendation(data, risk)

    conf_str = f"{confidence:.2f}%" if isinstance(confidence, (int, float)) else str(confidence)

    lines = []
    lines.append("=" * 52)
    lines.append("SMARTGRID SENTINEL REPORT")
    lines.append("=" * 52)
    lines.append("")
    lines.append("Prediction Model")
    lines.append("-" * 16)
    lines.append("Bidirectional LSTM")
    lines.append("")
    lines.append("Location")
    lines.append("-" * 16)
    lines.append(upazila)
    lines.append(f"{district} District")
    if division:
        lines.append(f"{division} Division")
    lines.append("")
    lines.append("Time")
    lines.append("-" * 16)
    lines.append(format_hour(hour))
    lines.append("")
    lines.append("Predicted Risk")
    lines.append("-" * 16)
    lines.append(risk.upper())
    lines.append("")
    lines.append("Confidence")
    lines.append("-" * 16)
    lines.append(conf_str)
    lines.append("")
    lines.append("AI Explanation")
    lines.append("-" * 16)
    for r in reasons:
        # Keep the explanation concise: strip trailing full-stop duplications
        bullet = r.rstrip('.')
        lines.append(f"- {bullet}.")
    lines.append("")
    lines.append("AI Recommendation")
    lines.append("-" * 16)
    for r in recs:
        lines.append(f"\u2713 {r}")
    lines.append("")
    lines.append("=" * 52)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: full explanation text for console / UI
# ---------------------------------------------------------------------------
def build_explanation_block(data, risk, confidence):
    reasons = generate_explanation(data)
    recs = generate_recommendation(data, risk)

    out = []
    out.append("AI EXPLANATION")
    out.append("The model predicts {} load shedding risk because:".format(risk.upper()))
    for r in reasons:
        out.append(f"  - {r}")
    out.append("")
    out.append("RECOMMENDATION")
    for i, r in enumerate(recs, 1):
        out.append(f"  {i}. {r}")
    return "\n".join(out)


if __name__ == "__main__":
    # Quick self-test
    sample = {
        'division_name': 'Sylhet',
        'district_name': 'Sylhet',
        'upazila_name': 'Golapganj',
        'temperature': 32.0,
        'humidity': 78.0,
        'rainfall': 5.0,
        'wind_speed': 12.0,
        'electricity_demand': 540.0,
        'generation_capacity': 430.0,
        'outages_24h': 3,
        'transformer_load': 97.0,
        'grid_stability': 38.0,
        'current_hour': 19,
    }
    print(format_report(sample, "High", "88-98%"))
    print()
    print(parse_natural_language_query(
        "Predict load shedding for Golapganj at 7 PM."))
    print(parse_natural_language_query(
        "Will Sylhet Sadar experience high risk tonight?"))