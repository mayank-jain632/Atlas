from pathlib import Path
import pandas as pd

class UniverseManager:

    def __init__(self, root="configs/universes"):
        self.root = Path(root)

    def load(self, name):
        df = pd.read_csv(self.root / f"{name}.csv")
        return df["Symbol"].dropna().tolist()

    def load_all(self):

        symbols = []

        print("Universe path:", self.root.resolve())
        print(list(self.root.iterdir()))
        for file in self.root.glob("*.csv"):

            print("Reading", file)

            df = pd.read_csv(file)

            print(df.head())

            symbols.extend(df["Symbol"].dropna().tolist())

        symbols = list(dict.fromkeys(symbols))

        print(f"Loaded {len(symbols)} unique symbols")

        return symbols