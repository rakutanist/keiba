# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
data = pd.read_pickle("data\shaped_data.pkl", compression="zip")

# %%
target = pd.get_dummies(data["着順"], prefix="着順").iloc[:, :3]
use_tag = [
    "LaceID",
    "競馬場",
    "重賞",
    "クラス",
    "コース",
    "馬場",
    "番",
    "体重",
    "年齢",
    "性別",
    "前走からの差分",
    "△",
    "▲",
    "☆",
    "★",
    "◇",
    "距離カテゴリ",
    "騎手1位率_全体",
    "騎手2位率_全体",
    "騎手3位率_全体",
    "騎手1位率競馬場コース",
    "騎手2位率競馬場コース",
    "騎手3位率競馬場コース",
    "騎手1位率脚質",
    "騎手2位率脚質",
    "騎手3位率脚質",
    "騎手1位率馬名",
    "騎手2位率馬名",
    "騎手3位率馬名",
    "父1位率_全体",
    "父2位率_全体",
    "父3位率_全体",
    "父1位率競馬場コース距離カテゴリ",
    "父2位率競馬場コース距離カテゴリ",
    "父3位率競馬場コース距離カテゴリ",
    "母父1位率_全体",
    "母父2位率_全体",
    "母父3位率_全体",
    "母父1位率競馬場コース距離カテゴリ",
    "母父2位率競馬場コース距離カテゴリ",
    "母父3位率競馬場コース距離カテゴリ",
    "馬名1位率_全体",
    "馬名2位率_全体",
    "馬名3位率_全体",
    "馬名1位率競馬場コース",
    "馬名2位率競馬場コース",
    "馬名3位率競馬場コース",
    "馬名1位率距離カテゴリ",
    "馬名2位率距離カテゴリ",
    "馬名3位率距離カテゴリ",
    "馬名1位率馬場",
    "馬名2位率馬場",
    "馬名3位率馬場",
    "先行",
    "差し",
    "追い込み",
    "逃げ",
    "出走間隔",
]
standerinze = ["体重", "前走からの差分", "出走間隔"]
use_data = data[use_tag]
use_data["前走からの差分"] = use_data["前走からの差分"].fillna(0)
max_len = use_data["番"].max()
# use_data = pd.get_dummies(use_data)
for i in standerinze:
    use_data[i] = (use_data[i] - use_data[i].mean()) / data[i].std()

# %%
id = use_data["LaceID"].unique()
list = []
unique_tag = [
    "LaceID",
    "競馬場",
    "重賞",
    "クラス",
    "コース",
    "馬場",
    "距離カテゴリ",
]
unique_df = pd.DataFrame(
    np.zeros((len(id), len(unique_tag))), columns=unique_tag, dtype=object
)
for i in tqdm(id):
    unique_df.iloc[i] = use_data[use_data["LaceID"] == i][unique_tag].values[0]
unique_df = unique_df.drop(columns="LaceID")
use_data = use_data.drop(columns=unique_tag)
use_data = pd.get_dummies(use_data)
use_data = pd.concat([use_data, target], axis=1)
unique_df = pd.get_dummies(unique_df)
# %%
result = np.zeros((len(id), 18 * 3))
feature = np.zeros((len(id), (18 * (use_data.shape[1] - 3))), dtype=object)
for i in tqdm(id):
    temp = use_data[data["LaceID"] == i].sort_values("番")
    score = temp[["着順_1", "着順_2", "着順_3"]].values
    if len(score) < 18:
        score = np.vstack([score, np.zeros((18 - len(score), 3))])
    temp = temp.drop(columns=["着順_1", "着順_2", "着順_3"])
    temparr = np.zeros((18, use_data.shape[1] - 3), dtype=object)
    for j in temp["番"]:
        temparr[j - 1] = temp[temp["番"] == j].values
    feature[i] = temparr.reshape(-1)
    result[i] = score.T.reshape(-1)

# %%
feature = np.hstack([unique_df.values, feature])
# %%
id_date = data.drop_duplicates(subset=["LaceID", "日付"])[["LaceID", "日付"]]
result = result / np.sum(result, axis=1).reshape(-1, 1)
np.savez("Preprocessed_data.npz", feature=feature, result=result, id_date=id_date)

# %%
