from datetime import datetime, timedelta

from config import START_DATE
from config import INTERVAL_HOURS


def generate_timeline(records=15000):

    timeline = []

    current = START_DATE

    for _ in range(records):
        timeline.append(current)
        current += timedelta(hours=INTERVAL_HOURS)

    return timeline


# Global timeline
timeline = generate_timeline()


if __name__ == "__main__":

    print("Timeline Length :", len(timeline))

    print()

    print("First 5")

    for t in timeline[:5]:

        print(t)

    print()

    print("Last 5")

    for t in timeline[-5:]:

        print(t)