from timeline import generate_timeline
from locations import LOCATIONS

timeline = generate_timeline()

upazilas = sum(len(v) for v in LOCATIONS.values())

print("=" * 50)
print("SMARTGRID DATASET CHECK")
print("=" * 50)

print(f"Timeline Points : {len(timeline)}")
print(f"Total Upazilas : {upazilas}")
print(f"Expected Dataset Rows : {len(timeline) * upazilas}")

print()

print("First Timestamp :", timeline[0])
print("Last Timestamp  :", timeline[-1])

print()

print("District Summary")

for district, ups in LOCATIONS.items():
    print(f"{district:15} : {len(ups)} Upazilas")