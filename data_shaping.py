# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm


# %%
def getstyle(corner):
    temp = np.array(list(corner)).astype(int)
    style = np.zeros(temp.shape[0], dtype=object)
    if len(temp[0]) != 0:
        arr = np.ceil(temp[:, -1] / (temp.shape[0] / 3))
        style[arr == 1] = "先行"
        style[arr == 2] = "差し"
        style[arr == 3] = "追い込み"
        style = np.where(np.any(temp == 1, axis=1), "逃げ", style)
    return style


def getage(arr):
    age = []
    gender = []
    for i in arr:
        if "せん" in i:
            gender.append("せん")
            age.append(int(i.replace("せん", "")))
        elif "牝" in i:
            gender.append("牝")
            age.append(int(i.replace("牝", "")))
        elif "牡" in i:
            gender.append("牡")
            age.append(int(i.replace("牡", "")))
        else:
            print("例外性別：", i)
    return age, gender


def getweight(arr):
    weight = []
    diff = []
    for i in arr:
        weight.append(int(i[: i.find("(")]))
        if ("初出走" in i) or ("前計不" in i):
            diff.append(None)
        else:
            diff.append(int((i[i.find("(") :]).replace("(", "").replace(")", "")))
    return [weight, diff]


# %%
raw_data = pd.read_pickle("data/raw.pkl", compression="zip")
horse = pd.read_pickle("data/horse_data.pkl", compression="zip")
result = raw_data["結果"]
raw_data = raw_data.drop(columns="結果")
raw_data["LaceID"] = raw_data.index
clm_raw = raw_data.columns.drop(["競馬場"]).to_list()
clm_result = result[0].columns.to_list()
# %%
list_result = result.tolist()
list_raw_data = raw_data.values.tolist()
# %%
# 2次元の配列に変更
savelist = []
style = []
for i, j in tqdm(list(zip(list_raw_data, list_result))):
    j[clm_raw] = i[0:1] + i[2:]
    j.drop(columns="馬URL")
    j = j[j["着順"] != "取消"]
    j = j[j["着順"] != "中止"]
    j = j[j["着順"] != "除外"]
    j = j[j["着順"] != "失格"]

    # j["脚質"] = getstyle(j["コーナー通過順位"])
    style.extend(getstyle(j["コーナー通過順位"]))
    savelist.extend(j.values.tolist())
# %%
data = pd.DataFrame(savelist, columns=clm_result + clm_raw)
data["脚質"] = style
# 年齢と性別を取得
data = data.assign(
    年齢=getage(data["馬齢"].values)[0], 性別=getage(data["馬齢"].values)[1]
)
# 場体重を現在の体重および前走からの差分に変更
data = data.assign(
    体重=getweight(data["馬体重"].values)[0],
    前走からの差分=getweight(data["馬体重"].values)[1],
)
# 不要なカラムを削除
data = data.drop(columns=["馬URL", "コーナー通過順位", "馬齢", "馬体重"])
data["日付"] = pd.to_datetime(data["日付"], format="%Y年%m月%d日")
data[["着順", "番"]] = data[["着順", "番"]].astype(int)

# %%
# 馬の父と母父の取得
h = horse.values
name = data["馬名"].values
pedigree = np.zeros((len(data), 2), dtype=object)
for i in tqdm(h):
    pedigree[name == i[0]] = i[[2, 3]]
data[["父", "母父"]] = pedigree
# %%
# 騎手の負担重量
arr = ["△", "▲", "☆", "★", "◇"]
for i in arr:
    temp = data["騎手"].str.contains(i)
    data[i] = temp
    data["騎手"] = data["騎手"].str.replace(i, "")
# %%
# 距離を距離カテゴリに変更
temp = data["距離"].values.astype(int)
cat_distance = np.where(temp < 1400, "短距離", "")
cat_distance = np.where((1400 <= temp) & (temp < 1800), "マイル", cat_distance)
cat_distance = np.where((1800 <= temp) & (temp < 2200), "中距離", cat_distance)
cat_distance = np.where((2200 <= temp) & (temp < 2800), "中長距離", cat_distance)
cat_distance = np.where(2800 <= temp, "長距離", cat_distance)
data["距離カテゴリ"] = cat_distance

# %%
jockey = data["騎手"].unique()
father = data["父"].unique()
grandfather = data["母父"].unique()
horse = data["馬名"].unique()
# %%
data["重賞"] = data["重賞"].mask(data["重賞"] == "ed", "リステッド")
data["重賞"] = data["重賞"].mask(data["重賞"] == "no", "重賞")
data["重賞"] = data["重賞"].mask(data["重賞"] == "n1", "jpn1")


# %%
@jit()
def numbafunc(arr):
    num_win = np.zeros((len(arr), 3))
    for k in range(len(arr)):
        num_win[k, 0] = np.count_nonzero(arr[:k] == 1)
        num_win[k, 1] = np.count_nonzero(arr[:k] == 2)
        num_win[k, 2] = np.count_nonzero(arr[:k] == 3)
    num_win = np.where(num_win != 0, num_win - 1, num_win)
    return num_win


def culc_winrate(data, tag, target=None, range=None):
    tag_idx = []
    if target is not None:
        if type(target) == list:
            for i in target:
                tag_idx.append(int(np.where(np.array(data.columns) == i)[0][0]))
        else:
            tag_idx.append(int(np.where(np.array(data.columns) == target)[0][0]))
    else:
        tag_idx = np.arange(len(data))
    ave_arr = np.zeros((len(data), 3))
    tag_list = data[tag].unique()
    for i in tqdm(tag_list):
        idx = data[tag] == i
        temp_df = data[idx]
        arr = temp_df.values
        arr2 = arr[:, 0].astype(int)
        if target is None:
            num_win = numbafunc(arr2)
            num_lace = np.arange(len(arr2))
            num_lace[0] = 1
            ave_arr[idx, :3] = num_win / num_lace.reshape(-1, 1)
        else:
            unique_combinations = temp_df.drop_duplicates(subset=target)[target].values
            # 条件ごとの勝率
            ave = np.zeros((len(arr2), 3))
            for j in unique_combinations:
                tempidx = np.all((arr[:, tag_idx] == j), axis=1)
                lace = np.count_nonzero(tempidx)
                num_lace = np.arange(lace)
                num_win = numbafunc(arr2[tempidx])
                num_lace[0] = 1
                ave[tempidx] = num_win / num_lace.reshape(-1, 1)
            ave_arr[idx] = ave
    return ave_arr


# %%
data = data.sort_values("日付")
rate_columns = pd.Series(["1位率", "2位率", "3位率"])
# 騎手の勝率を計算
tags = [["競馬場", "コース"], "脚質", "馬名"]
factor = "騎手"
jockey_winrate = culc_winrate(data, factor)
col = factor + rate_columns + "_全体"
data[col.values] = jockey_winrate
for i in tags:
    winrate = culc_winrate(data, factor, i)
    string = ""
    if len(i) > 1:
        for j in i:
            string = string + j
        col = factor + rate_columns + string
    else:
        col = factor + rate_columns + i
    data[col.values] = winrate
# %%
# 父産馬の勝率を計算
tags = [["競馬場", "コース", "距離カテゴリ"]]
factor = "父"
jockey_winrate = culc_winrate(data, factor)
col = factor + rate_columns + "_全体"
data[col.values] = jockey_winrate
for i in tags:
    winrate = culc_winrate(data, factor, i)
    string = ""
    if len(i) > 1:
        for j in i:
            string = string + j
        col = factor + rate_columns + string
    else:
        col = factor + rate_columns + i
    data[col.values] = winrate
# %%
# 母父産馬の勝率を計算
tags = [["競馬場", "コース", "距離カテゴリ"]]
factor = "母父"
jockey_winrate = culc_winrate(data, factor)
col = factor + rate_columns + "_全体"
data[col.values] = jockey_winrate
for i in tags:
    winrate = culc_winrate(data, factor, i)
    string = ""
    if len(i) > 1:
        for j in i:
            string = string + j
        col = factor + rate_columns + string
    else:
        col = factor + rate_columns + i
    data[col.values] = winrate
# %%
# 競走馬のこれまでの勝率を計算
tags = [["競馬場", "コース"], "距離カテゴリ", "馬場"]
factor = "馬名"
jockey_winrate = culc_winrate(data, factor)
col = factor + rate_columns + "_全体"
data[col.values] = jockey_winrate
for i in tags:
    winrate = culc_winrate(data, factor, i)
    string = ""
    if len(i) > 1:
        for j in i:
            string = string + j
        col = factor + rate_columns + string
    else:
        col = factor + rate_columns + i
    data[col.values] = winrate
# %%
# 競走馬のこれまでの脚質を取得
style_dummy = pd.get_dummies(data["脚質"]).iloc[:, 1:]
style_rate = np.zeros_like(style_dummy.values, dtype=float)
for i in tqdm(horse):
    idx = data["馬名"] == i
    temp_style = style_dummy[idx].values
    rate_temp = np.zeros_like(temp_style, dtype=float)
    for j in range(1, len(temp_style)):
        rate_temp[j] = np.count_nonzero(temp_style[:j], axis=0) / j
    style_rate[idx] = rate_temp
data[style_dummy.columns] = style_rate

# %%
# 前走からの期間を取得
interval = pd.Series(np.zeros(len(data)), dtype="timedelta64[s]")
for i in tqdm(horse):
    idx = data["馬名"] == i
    temp = data[idx]
    # interval_temp = pd.Series(np.zeros(len(temp)), dtype="timedelta64[s]")
    interval_temp = temp["日付"].iloc[1:].values - temp["日付"].iloc[:-1].values
    interval_temp = np.hstack([0, interval_temp])
    # for j in range(1, len(temp)):
    #     interval_temp.iat[j] = temp["日付"].iat[j] - temp["日付"].iat[j - 1]
    interval[idx] = interval_temp
data["出走間隔"] = interval
data["出走間隔"] = data["出走間隔"].dt.days
# %%
data["年齢"] = data["年齢"].astype(str)
data.to_pickle("data/shaped_data.pkl", compression="zip")

# %%
