# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %%
prefix = "https://www.jra.go.jp/"
target = Path("./URL")
save_df = pd.DataFrame(columns=["lace", "URL"])
for i in target.glob("URL_child*"):
    with open(i) as f:
        line = f.readline().split(sep=",")
    df = pd.DataFrame(columns=["lace", "URL"])
    df["URL"] = line
    df["URL"] = prefix + df["URL"]
    df["lace"] = i.name.replace("URL_child_", "").replace(".csv", "")
    save_df = pd.concat([save_df, df], axis=0, ignore_index=True)

save_df.to_pickle(target / "URL.pkl", compression="zip")
# %%
