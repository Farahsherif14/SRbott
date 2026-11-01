"""Generate the mock unit registry (compounds, buildings, unit numbers)."""
import pandas as pd
from config import PROC

COMPOUNDS = ["Palm Gardens", "Nile View", "Sunrise Heights", "Marina Park", "Garden Court"]


def main(n_buildings=6, floors=8, per_floor=4):
    rows = []
    for comp in COMPOUNDS:
        for b in range(n_buildings):
            bld = chr(ord("A") + b)
            for f in range(1, floors + 1):
                for u in range(1, per_floor + 1):
                    rows.append({"compound": comp, "building": bld, "unit": f"{bld}-{f}{u:02d}"})
    df = pd.DataFrame(rows)
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC / "mock_units.csv", index=False)
    print(f"Wrote {len(df)} units -> data/processed/mock_units.csv")


if __name__ == "__main__":
    main()