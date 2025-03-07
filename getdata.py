# %%
import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# %%
data_tags = [
    "日付",
    "競馬場",
    "重賞",
    "クラス",
    "コース",
    "距離",
    "方向",
    "天気",
    "馬場",
    "結果",
]
tags = [
    "着順",
    "枠",
    "番",
    "馬名",
    "馬URL",
    "馬齢",
    "体重",
    "騎手",
    "馬体重",
    "調教師",
    "コーナー通過順位",
    "タイム",
    "3F",
]

df_URL = pd.read_pickle("URL/URL.pkl", compression="zip")
save_df = pd.DataFrame(
    np.zeros((len(df_URL), len(data_tags))), columns=data_tags, dtype=object
)
for j, i in enumerate(tqdm(df_URL.values)):
    keibajo = i[0][i[0].rfind("回") + 1 : i[0].rfind("回") + 3]
    URL = i[1]
    # HTMLの取得(GET)
    req = requests.get(URL)
    req.encoding = req.apparent_encoding  # 日本語の文字化け防止
    bsObj = BeautifulSoup(req.text, "html.parser")
    # 要素の抽出
    baba = bsObj.find_all("div", {"class": "cell baba"})[0]
    baba_data = baba.get_text(separator=" ", strip=True).split()[1:]
    lace_category = bsObj.find_all("div", {"class": "cell class"})[0].get_text(
        separator=" ", strip=True
    )
    course = bsObj.find_all("div", {"class": "cell course"})[0].get_text(
        separator=" ", strip=True
    )
    distance = (
        course[: course.rfind("メートル")].replace("コース：", "").replace(",", "")
    )
    direction = course[course.rfind("・") + 1]

    date = bsObj.find_all("div", {"class": "cell date"})[0].get_text(
        separator=" ", strip=True
    )
    idx = re.search(
        "（",
        date,
    )
    date = date[: idx.start()]

    place = bsObj.find_all("td", {"class": "place"})
    waku = bsObj.find_all("td", {"class": "waku"})
    num = bsObj.find_all("td", {"class": "num"})
    horse = bsObj.find_all("td", {"class": "horse"})
    age = bsObj.find_all("td", {"class": "age"})
    weight = bsObj.find_all("td", {"class": "weight"})
    jockey = bsObj.find_all("td", {"class": "jockey"})
    h_weight = bsObj.find_all("td", {"class": "h_weight"})
    trainer = bsObj.find_all("td", {"class": "trainer"})
    corner = bsObj.find_all("td", {"class": "corner"})
    jusho = bsObj.find_all("span", {"class": "grade_icon"})
    time_total = bsObj.find_all("td", {"class": "time"})
    time_3F = bsObj.find_all("td", {"class": "f_time"})
    save_df["競馬場"].iat[j] = keibajo
    save_df["クラス"].iat[j] = lace_category
    save_df["天気"].iat[j] = baba_data[0]
    save_df["コース"].iat[j] = baba_data[1]
    save_df["馬場"].iat[j] = baba_data[2]
    save_df["日付"].iat[j] = date
    save_df["距離"].iat[j] = distance
    save_df["方向"].iat[j] = direction
    if len(jusho) != 0:
        temp = jusho[0].find("img")["src"]
        save_df["重賞"].iat[j] = temp[temp.rfind(".png") - 2 : temp.rfind(".png")]
    else:
        save_df["重賞"].iat[j] = "なし"

    temp_df = pd.DataFrame(np.zeros((len(waku), len(tags))), columns=tags, dtype=object)
    for i in range(len(place)):
        # 着順
        temp_df["着順"].iat[i] = place[i].get_text()

        # 枠
        pos_posfix = re.search(
            ".png",
            str(waku[i]),
        )
        pos_prefix = re.search(
            "waku/",
            str(waku[i]),
        )
        temp_df["枠"].iat[i] = str(waku[i])[pos_prefix.end() : pos_posfix.start()]
        # 番
        temp_df["番"].iat[i] = num[i].get_text(strip=True)
        # 馬名
        temp_df["馬名"].iat[i] = horse[i].get_text(strip=True)
        # 馬URL
        temp_df["馬URL"].iat[i] = horse[i].find("a")["href"]
        # 馬齢
        temp_df["馬齢"].iat[i] = age[i].get_text(strip=True)
        # 体重
        temp_df["体重"].iat[i] = weight[i].get_text(strip=True)
        # 騎手
        temp_df["騎手"].iat[i] = jockey[i].get_text(strip=True)
        # 馬体重
        temp_df["馬体重"].iat[i] = h_weight[i].get_text(strip=True)
        # 調教師
        temp_df["調教師"].iat[i] = trainer[i].get_text(strip=True)
        # コーナー追加順位
        temp_df["コーナー通過順位"].iat[i] = (
            corner[i].get_text(strip=True, separator=" ").split()
        )
        # タイム
        temp_df["タイム"].iat[i] = time_total[i].get_text(strip=True)
        # 3F
        temp_df["3F"].iat[i] = time_3F[i].get_text(strip=True)

    save_df["結果"].iat[j] = temp_df
    # DDOSじゃないようにする
    time.sleep(0.5)
save_df.to_pickle("data/raw.pkl", compression="zip")
# %%
