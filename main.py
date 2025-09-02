# inspired by: https://foolishfox.cn/posts/202402-WeChatMsgAnalysis.html

import sys
import os
import re
import time
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from paddlenlp import Taskflow
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

MESSAGE_TYPES = ['text', 'image', 'audio', 'video', 'deleted', 'call']
punctuation = '.,?!'

with open('stop_words.txt', 'r') as f:
    STOP_WORDS = f.read().split('\n')

def parse_lines(file_name:str, 
                start_time:time.struct_time, 
                end_time:time.struct_time):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.read()

    pattern = r"\[(.*?)\] (.*?): (.*?)\n"
    match = re.findall(pattern, lines)
    if match:
        m_times = []
        names = []
        messages = []
        m_types = []
        for line in match:
            m_time = time.strptime(line[0], '%Y/%m/%d, %H:%M:%S')
            if m_time < start_time or m_time > end_time:
                continue
            
            is_text = False
            if 'image omitted' in line[2]:
                m_types.append('image')
            elif 'audio omitted' in line[2]:
                m_types.append('audio')
            elif 'video omitted' in line[2]:
                m_types.append('video')
            elif ('This message was deleted.' in line[2] or 
                'You deleted this message.' in line[2]):
                m_types.append('deleted')
            elif ('Messages and calls are' in line[2] or 'Missed voice call' in line[2]):
                m_types.append('call')
            else:
                m_types.append('text')
                is_text = True
            
            if is_text:
                m_times.append(m_time)
                names.append(line[1])
                messages.append(line[2])
            else:
                m_times.append(m_time)
                names.append(line[1])
                messages.append('')
        
        data = pd.DataFrame({'time': m_times, 
                            'name': names, 
                            'message': messages, 
                            'type': m_types})
        return data
    else:
        return None

def frequency_analyse(df:pd.DataFrame, labels:list, save_name:str):
    data = {}
    for name in labels:
        data_one_person = []
        for m_type in MESSAGE_TYPES:
            data_one_person.append(len(df[df['name'] == name].query("type == @m_type")))
        data[name] = data_one_person

    data = (
        pd.DataFrame(data, index = MESSAGE_TYPES)
        .reset_index()
        .melt("index")
        .rename(columns = {"index": "type", "variable": "person", "value": "count"})
    )
    g = sns.catplot(data, kind = "bar", x = "type", y = "count", hue = "person", 
                    palette = "dark", alpha = 0.6, height = 6)

    for ax in g.axes.ravel():
        for i in range(2):
            ax.bar_label(ax.containers[i], fontsize = 9)
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.8, 0.92))
    plt.yscale("log")

    g.figure.set_size_inches(6, 5)
    g.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def message_length_analyse(df:pd.DataFrame, labels:list, save_name:str):
    sN = 3
    multiple = "dodge"

    mu, std = 0, 0
    data = {"length": [], "person": []}
    for name in labels:
        length = df[(df['name'] == name) & 
                    (df['type'] == 'text')]['message'].apply(len).tolist()
        data["length"] += length
        data["person"] += [name] * len(length)
        if np.mean(length) + sN * np.std(length) > mu + std:
            mu, std = np.mean(length), np.std(length)
    xlim = int(np.ceil(mu + sN * std))

    data = pd.DataFrame(data)
    bins = np.linspace(0, xlim, xlim + 1)

    ax = sns.histplot(
        data = data,
        x = "length",
        hue = "person",
        bins = bins,
        multiple = multiple,
        edgecolor = ".3",
        linewidth = 0.5,
        palette = "dark",
        alpha = 0.6,
    )
    ax.set_xlim(0, xlim)
    ax.set_xlabel("Length of Message")

    ax.figure.set_size_inches(8, 5)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def time_in_day_analyse(df:pd.DataFrame, labels:list, save_name:str):
    multiple = "dodge"

    data = {"time": [], "person": []}
    for name in labels:
        hour = df[df['name'] == name]['time'].apply(lambda x: x.tm_hour).tolist()
        data["time"] += hour
        data["person"] += [name] * len(hour)

    data = pd.DataFrame(data)
    bins = np.arange(0, 25, 1)

    ax = sns.histplot(
        data = data,
        x = "time",
        hue = "person",
        bins = bins,
        multiple = multiple,
        edgecolor = ".3",
        linewidth = 0.5,
        palette = "dark",
        alpha = 0.6,
    )
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Hour")
    ax.set_xlim(0, 24)
    sns.move_legend(ax, loc = "upper center", bbox_to_anchor = (0.5, 1.2), ncol = 2)

    ax.figure.set_size_inches(8, 5)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def time_in_week_analyse(df:pd.DataFrame, save_name:str):
    df['day'] = df['time'].apply(lambda x: x.tm_wday)
    grouper = pd.Grouper(key = "day")
    data = df.groupby(grouper).count()['name']
    data = data.sort_index()
    data.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    ax = sns.barplot(data=data, errorbar=None)
    ax.set_xlabel("Weekday")
    ax.bar_label(ax.containers[0], fontsize=10)

    ax.figure.set_size_inches(5, 5)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def time_in_year_analyse(df:pd.DataFrame, year:int, save_name:str):
    wTicks = 500
    global wStart
    global wEnd

    grouper = pd.Grouper(key = "datetime", freq = "W-MON")
    data = df.groupby(grouper).count()['name'].to_frame()
    data = data.reindex(pd.date_range(start = wStart, end = wEnd, freq = "W-MON"))
    data.index = pd.date_range(start = wStart, end = wEnd, freq = "W-MON").strftime("%m-%d")
    data = data.fillna(0)
    data.columns = ["Count"]

    vM = np.ceil(data["Count"].max() / wTicks) * wTicks
    norm = plt.Normalize(0, vM)
    sm = plt.cm.ScalarMappable(cmap = "Reds", norm = norm)

    ax = sns.barplot(x = data.index, y = data["Count"], hue = data["Count"], 
                    hue_norm = norm, palette = "Reds")
    ax.set_xlabel("Date")
    plt.xticks(rotation = 60)
    for bar in ax.containers:
        ax.bar_label(bar, fontsize = 10, fmt = "%.0f")
    ax.get_legend().remove()

    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(axpos.x1 + 0.02, axpos.y0, axpos.x1 + 0.03, axpos.y1)
    cax = ax.figure.add_axes(caxpos)

    formatter = mticker.StrMethodFormatter("{x:.0f}")
    cax.figure.colorbar(sm, cax = cax, format = formatter)

    ax.figure.set_size_inches(20, 8)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def enthusiasm_analyse(df:pd.DataFrame, labels:list, year:int, save_name:str):
    global wStart
    global wEnd
    
    grouper = pd.Grouper(key = "datetime", freq = "W-MON")
    df_W1 = df[df['name'] == labels[0]].groupby(grouper).count()['name']
    df_W2 = df[df['name'] == labels[1]].groupby(grouper).count()['name']

    data = pd.DataFrame({"E": (df_W1 - df_W2) / (df_W1 + df_W2)})
    data = data.reindex(pd.date_range(start = wStart, end = wEnd, freq = "W-MON"))
    data.index = pd.date_range(start = wStart, end = wEnd, freq = "W-MON").strftime("%m-%d")
    data = data.fillna(0)

    vM = data["E"].abs().max()
    norm = plt.Normalize(-vM, vM)
    sm = plt.cm.ScalarMappable(cmap = "coolwarm", norm = norm)
    
    ax = sns.barplot(x = data.index, y = data["E"], hue = data["E"], 
                    hue_norm = norm, palette = "coolwarm")
    ax.set_xlabel("Date")
    plt.xticks(rotation = 60)
    ax.set_ylabel("Enthusiasm Index")
    for bar in ax.containers:
        ax.bar_label(bar, fontsize = 10, fmt = "%.2f")
    ax.get_legend().remove()

    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(axpos.x1 + 0.02, axpos.y0, axpos.x1 + 0.03, axpos.y1)
    cax = ax.figure.add_axes(caxpos)

    locator = mticker.MultipleLocator(0.2)
    formatter = mticker.StrMethodFormatter("{x:.1f}")
    cax.figure.colorbar(sm, cax = cax, ticks = locator, format = formatter)

    ax.figure.set_size_inches(20, 8)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def year_map(df:pd.DataFrame, year:int, save_name:str):
    global wStart
    global wEnd

    grouper = pd.Grouper(key = "datetime", freq = "D")
    data = df.groupby(grouper).count()['name']
    data = data.to_frame()
    data.columns = ["Count"]

    data["date"] = data.index
    data["week"] = data["date"].dt.isocalendar()["week"]
    data["day"] = data["date"].dt.dayofweek
    data.index = range(len(data))
    for i in range(7):
        if data.loc[i, "week"] > 1:
            data.loc[i, "week"] = 0

    data = data.pivot(index = "day", columns = "week", values = "Count")
    data.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    data.columns = pd.date_range(start = wStart, end = wEnd, freq = "W-MON").strftime("%m-%d")

    ax = sns.heatmap(
        data,
        annot = False,
        linewidths = 0.5,
        cbar_kws = {"orientation": "vertical", "location": "left", "pad": 0.03},
        cmap = "Blues",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekday")
    ax.figure.set_size_inches(24, 4)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()

def word_cloud(text:str, save_name:str, mask_image:str = None):
    if not mask_image is None:
        mask_image = np.array(Image.open(mask_image))
    wc = WordCloud(
        background_color = "white",
        max_words = 200,
        max_font_size = 100,
        random_state = 42,
        width = 800,
        height = 400,
        mask = mask_image,
    ).generate(text)

    plt.figure(figsize=(20,10))
    plt.imshow(wc, interpolation = "bilinear")
    plt.axis("off")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()

def high_frequency_words(name_1_words:list, name_2_words:list, total_words:list, labels:list, save_name:str):
    wN = 50

    word_freq = dict(Counter(total_words).most_common(wN))

    word_freq_1 = []
    word_freq_2 = []
    for word in word_freq.keys():
        word_freq_1.append(name_1_words.count(word))
        word_freq_2.append(name_2_words.count(word))

    data = pd.DataFrame(
        {
            "words": list(word_freq.keys()),
            labels[0]: word_freq_1,
            labels[1]: word_freq_2,
            "sum": [word_freq_1[i] + word_freq_2[i] for i in range(len(word_freq_1))],
        }
    )

    grouper = pd.Grouper(key = "words")
    data = data.groupby(grouper).sum()
    data = data.sort_values(by = "sum", ascending = False)

    ratio = data[labels[0]] / data["sum"]
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap = "coolwarm", norm = norm)

    fig = plt.figure(figsize = (10, 10), dpi = 300)
    grid = plt.GridSpec(1, 4, wspace = 0.5)

    ax0 = fig.add_subplot(grid[0, 0])
    sns.barplot(x = -data[labels[0]], y = data.index, ax = ax0, hue = ratio, 
                hue_norm = norm, palette = "coolwarm")
    plt.xticks(rotation = 45)
    ax1 = fig.add_subplot(grid[0, 1:])
    sns.barplot(x = data[labels[1]], y = data.index, ax = ax1, hue = (1 - ratio), 
                hue_norm = norm, palette = "coolwarm")
    plt.xticks(rotation = 45)

    scale = 300
    ax0.set_xlabel("frequency")
    ax0.set_ylabel("")
    ax0.set_xticks(range(int(-np.ceil(max(word_freq_1)/scale)*scale), 
                        int(-np.floor(min(word_freq_1)/scale)*scale) + 1, scale))
    ax0.set_yticks([])
    ax0.spines["left"].set_visible(False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set_title(labels[0])
    ax0.get_legend().remove()

    ax1.set_xlabel("frequency")
    ax1.set_ylabel("")
    ax1.set_xticks(range(int(np.floor(min(word_freq_2)/scale)*scale), 
                        int(np.ceil(max(word_freq_2)/scale)*scale) + 1, scale))
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title(labels[1])
    ax1.get_legend().remove()

    axpos = ax1.get_position()
    caxpos = mtransforms.Bbox.from_extents(axpos.x0 + 0.06, axpos.y0 + 0.03, axpos.x1, axpos.y0 + 0.04)
    cax = ax1.figure.add_axes(caxpos)

    locator = mticker.MultipleLocator(0.1)
    formatter = mticker.StrMethodFormatter("{x:.1f}")
    cax.figure.colorbar(sm, cax = cax, orientation = "horizontal", ticks = locator, format = formatter)
    cax.set_title("ratio")

    x0 = ax0.get_position().x1
    x1 = ax1.get_position().x0
    xm = (x0 + x1) / 2
    y0 = ax0.get_position().y0
    y1 = ax0.get_position().y1

    for i in range(wN):
        fig.text(
            xm, y0 + (y1 - y0) * (wN - i - 0.5) / wN, data.index[i],
            color="black", ha="center", va="center"
        )

    fig.set_dpi(150)
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()

def get_sentiment_data(df:pd.DataFrame):
    dfE = df[df['type'] == 'text']
    dfE.index = range(len(dfE))

    senta = Taskflow("sentiment_analysis")
    scores = pd.DataFrame(senta([i for i in dfE["message"].to_list()]))
    scores.loc[scores["label"] == "negative", "score"] = 1 - scores.loc[scores["label"] == "negative", "score"]

    dfE["score"] = scores["score"]
    dfE["score"] = 2 * dfE["score"] - 1

    return dfE

def sentiment_analyse(dfE:pd.DataFrame, save_name:str):
    ax = sns.histplot(data = dfE, x = "score", hue = "name", palette = "dark", alpha = 0.6, bins = 100)

    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Distribution")
    ax.set_xlim(-1, 1)

    ax.figure.set_size_inches(8, 3)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def sentiment_in_year_analyse(dfE:pd.DataFrame, year:int, save_name:str):
    global wStart
    global wEnd

    grouper = pd.Grouper(key="datetime", freq="W-MON")
    data = dfE.groupby(grouper)["score"].mean().to_frame()
    data = data.reindex(pd.date_range(start = wStart, end = wEnd, freq = "W-MON"))
    data.index = pd.date_range(start = wStart, end = wEnd, freq = "W-MON").strftime("%m-%d")
    data = data.fillna(0)
    data.columns = ["score"]

    vM = data["score"].abs().max()
    norm = plt.Normalize(-vM, vM)
    sm = plt.cm.ScalarMappable(cmap = "coolwarm", norm = norm)

    ax = sns.barplot(x = data.index, y = data["score"], hue = data["score"], 
                    hue_norm = norm, palette = "coolwarm")
    ax.set_xlabel("Date")
    plt.xticks(rotation = 60)
    for bar in ax.containers:
        ax.bar_label(bar, fontsize = 10, fmt = "%.2f")
    ax.get_legend().remove()

    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(axpos.x1 + 0.02, axpos.y0, axpos.x1 + 0.03, axpos.y1)
    cax = ax.figure.add_axes(caxpos)

    # locator = mticker.MultipleLocator(1)
    formatter = mticker.StrMethodFormatter("{x:.2f}")
    cax.figure.colorbar(sm, cax = cax, format = formatter)

    ax.figure.set_size_inches(20, 8)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

    return data["score"]

def sentiment_change_analyse(avgSenScore:pd.DataFrame, labels:list, save_name:str):
    ax = sns.lineplot(data = avgSenScore[labels[0]], linewidth = 3, 
                    marker = "s", markersize = 15, label = labels[0])
    ax = sns.lineplot(data = avgSenScore[labels[1]], linewidth = 3, 
                    marker = "^", markersize = 15, ax = ax, label = labels[1])

    ax.set_xlabel("Date")
    plt.xticks(rotation = 60)
    ax.set_ylabel("Average Sentiment Score")
    ax.set_xlim(0, 52)
    ax.legend(prop = {"size": 24})

    ax.figure.set_size_inches(20, 8)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def sentiment_in_day_analyse(dfE:pd.DataFrame, labels:list, save_name:str):
    multiple = "dodge"

    dfE['hour'] = dfE['time'].apply(lambda x: x.tm_hour)
    grouper = pd.Grouper(key="hour")

    data = []
    for name in labels:
        tmp = dfE[dfE['name'] == name].groupby(grouper)["score"].mean().sort_index()
        for i in range(24):
            if i in tmp.index:
                data.append(tmp[i])
            else:
                data.append(0)
        data.append(0)
    data = pd.DataFrame(
        {
            "score": data,
            "person": [labels[0]] * 25 + [labels[1]] * 25,
        }
    )

    xBins = [i for i in range(25)]
    ax = sns.histplot(
        data = data,
        x = xBins * 2,
        bins = xBins,
        weights = "score",
        hue = "person",
        multiple = multiple,
        edgecolor = ".3",
        linewidth = 0.5,
        palette = "dark",
        alpha = 0.6,
    )

    ax.set_xticks(range(25))
    ax.set_xticklabels(range(25))
    ax.set_xlabel("Hour")
    ax.set_xlim(0, 24)
    ax.set_ylim(np.min([0, np.floor(data["score"].min() / 0.05) * 0.05]), np.ceil(data["score"].max() / 0.05) * 0.05)
    sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)

    ax.figure.set_size_inches(8, 4)
    ax.figure.set_dpi(150)
    plt.savefig(save_name, bbox_inches = 'tight')
    plt.close()

def main(CHAT_FILE, SAVE_FOLDER, name_in_file_1, name_in_file_2, name_1, name_2, year):
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    
    sns.set_style("darkgrid")
    labels = [name_1, name_2]

    start_time = year + '/01/01, 00:00:00'
    end_time = year + '/12/31, 23:59:59'
    year = int(year)
    global wStart
    global wEnd
    wStart = str(year) + '-01-02'
    wEnd = str(year + 1) + '-01-05'
    print(f'Parsing data from{start_time} to {end_time}...')
    start_time = time.strptime(start_time, '%Y/%m/%d, %H:%M:%S')
    end_time = time.strptime(end_time, '%Y/%m/%d, %H:%M:%S')

    df = parse_lines(CHAT_FILE, start_time, end_time)
    print(f'{len(df)} messages sent')
    df['datetime'] = df['time'].apply(lambda x: datetime(*x[:6]))
    df['name'] = df['name'].replace({name_in_file_1: name_1, name_in_file_2: name_2})
    
    words_1_list = df[df['name'] == name_1]['message'].tolist()
    words_2_list = df[df['name'] == name_2]['message'].tolist()
    words_1_list = ' '.join(words_1_list).split(' ')
    words_2_list = ' '.join(words_2_list).split(' ')
    translator = str.maketrans('', '', punctuation)
    words_1_list = [word.lower().translate(translator) for word in words_1_list]
    words_2_list = [word.lower().translate(translator) for word in words_2_list]
    words_1_list = [word for word in words_1_list 
                    if (word not in STOP_WORDS) and (word != '')]
    words_2_list = [word for word in words_2_list 
                    if (word not in STOP_WORDS) and (word != '')]
    words_total_list = words_1_list + words_2_list
    words_1 = ' '.join(words_1_list)
    words_2 = ' '.join(words_2_list)
    words_total = ' '.join(words_total_list)
    
    pbar = tqdm(total = 17)
    frequency_analyse(df, labels, SAVE_FOLDER + 'frequency.png')
    pbar.update(1)
    message_length_analyse(df, labels, SAVE_FOLDER + 'message_length.png')
    pbar.update(1)
    time_in_day_analyse(df, labels, SAVE_FOLDER + 'time_in_day.png')
    pbar.update(1)
    time_in_week_analyse(df, SAVE_FOLDER + 'time_in_week.png')
    pbar.update(1)
    time_in_year_analyse(df, year, SAVE_FOLDER + 'time_in_year.png')
    pbar.update(1)
    enthusiasm_analyse(df, labels, year, SAVE_FOLDER + 'enthusiasm.png')
    pbar.update(1)
    year_map(df, year, SAVE_FOLDER + 'year_map.png')
    pbar.update(1)
    word_cloud(words_total, SAVE_FOLDER + 'word_cloud_with_mask.png', 'mask.jpg')
    pbar.update(1)
    word_cloud(words_total, SAVE_FOLDER + 'word_cloud.png') # without mask
    pbar.update(1)
    word_cloud(words_1, SAVE_FOLDER + f'word_cloud_{labels[0]}.png')
    pbar.update(1)
    word_cloud(words_2, SAVE_FOLDER + f'word_cloud_{labels[1]}.png')
    pbar.update(1)
    high_frequency_words(words_1_list, words_2_list, words_total_list, labels, SAVE_FOLDER + 'high_frequency_words.png')
    pbar.update(1)
    dfE = get_sentiment_data(df)
    sentiment_analyse(dfE, SAVE_FOLDER + 'sentiment.png')
    pbar.update(1)
    avgSenScore = {}
    for name in labels:
        avgSenScore[name] = sentiment_in_year_analyse(dfE[dfE['name'] == name], 
                                                year, 
                                                SAVE_FOLDER + f'sentiment_in_year_{name}.png')
        pbar.update(1)
    sentiment_change_analyse(avgSenScore, labels, SAVE_FOLDER + 'sentiment_change.png')
    pbar.update(1)
    sentiment_in_day_analyse(dfE, labels, SAVE_FOLDER + 'sentiment_in_day.png')
    pbar.update(1)
    pbar.close()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])