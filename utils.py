'''
    Funções úteis para a análise de classificadores
'''
import warnings
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import seaborn as sns

import pandas as pd
import numpy as np
from scipy.stats import norm

# retorna um dict contendo media, variancia e limites inf & sup de uma lista de scores
def get_score_stts(scores):
    r = {}
    r["mean"], r["std"] = scores.mean(), scores.std()

    scale = r["std"]/np.sqrt(len(scores))
    with warnings.catch_warnings():
        if r["std"] == 0.0:
            warnings.simplefilter('ignore') #Ignora warning que são causados quando a variância = 0
        r["inf"], r["sup"] = np.nan_to_num(norm.interval(0.95, loc=r["mean"], scale=scale), nan=r["mean"])
        
    return r

# Plota a matrix de confusão usando Heatmap do seaborn
def plot_conf_mat(conf_mat, labels, title):
    df_cmap = pd.DataFrame(conf_mat, index = labels, columns = labels )

    sns.set(font_scale=1.4) # for label size
    _, ax = plt.subplots(figsize=(6,5)) 

    color_pallet = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    sns.heatmap(df_cmap, cmap=color_pallet, annot=True,ax=ax, fmt='d')
    ax.set(ylabel='Verdadeiro', xlabel='Previsto', title=title)
    # plt.yticks(rotation=20)
    plt.tick_params(axis='both', which='major', labelsize=14, labelbottom = False, left=True, top = True, labeltop=True)



def plot_dataset_boxplot(dataset_df, dataset_name):
    # boxplot zscores and tempos
    figsize = (8, 5)
    _, ax = plt.subplots(figsize=figsize)
    ax.set(xlabel='Acurácia', ylabel='Classificador')

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    sns.boxplot(data=dataset_df.T, ax=ax, orient="h", palette="Set3")
    ax.set_title(dataset_name)
    # plt.yticks(rotation=20)
    plt.tight_layout()