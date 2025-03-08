# %%
import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# %%
raw_data = pd.read_pickle("data/raw.pkl", compression="zip")

# %%
result = []
# %%
for i in tqdm(raw_data["結果"].values):
    for j in i.values:
        result.append(j.tolist())


# %%
result = pd.DataFrame(result, columns=raw_data["結果"].iat[0].columns)
horse = result[["馬名", "馬URL"]].drop_duplicates(subset="馬名")
pedigree = np.zeros((len(horse), 4), dtype=object)
prefix = "https://www.jra.go.jp/"
for k, i in enumerate(tqdm(horse["馬URL"])):
    # HTMLの取得(GET)
    req = requests.get(prefix + i)
    req.encoding = req.apparent_encoding  # 日本語の文字化け防止
    bsObj = BeautifulSoup(req.text, "html.parser")
    # 要素の抽出
    data = bsObj.find_all(class_="data_col1")
    for j in data:
        if j.find("dt").text == "父":
            father = j.find("dd").text
        elif j.find("dt").text == "母の父":
            grandfather = j.find("dd").text
    data = bsObj.find_all(class_="data_col3")
    for j in data:
        if j.find("dt").text == "馬主名":
            orner = j.find("dd").text
        elif j.find("dt").text == "生産牧場":
            bokujo = j.find("dd").text

    pedigree[k] = [father, grandfather, orner, bokujo]
    time.sleep(0.3)
    # %%
horse[["父", "母父"]] = pedigree
horse.to_pickle("horse_data.pkl", compression="zip")
# %%
