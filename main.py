
# %% [markdown]
# ### get_score_stts(scores):
# ####    Entrada: lista de scores
# ####    Retorno: Dicionário de informações (statuses)
#     {Média, desvio padrão, lim inferior, lim superior}

# %%
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_conf_mat, plot_dataset_boxplot
from kcentroides import Kcentroides, kga, kmeans
from etapas import phase_one, phase_two
from one_r_probabilistico import OneRProbabilistico



# %% [markdown]
# #### RANDOM SEED

# %%
rand_state = 36851234

# %% [markdown]
# #### Exemplo Fase 1:

# %%
from sklearn import datasets
OneR = OneRProbabilistico(random_state=rand_state)
phase_one(datasets.load_iris(), OneR, needs_discretizer=True, random_state=rand_state)


# %% [markdown]
# #### Exemplo etapa 2:

# %%
from sklearn import datasets
kmeansC = Kcentroides(method=kmeans, random_state=rand_state)
grid = {'estimator__k': [1,3,5]}
phase_two(datasets.load_iris(), kmeansC, grid, random_state=rand_state)

# %% [markdown]
# ### INICIALIZANDO CLASSIFICADORES E DATASETS

# %%
from sklearn import datasets

# Classificadores
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rand_state = 36851234

datasets_ = {
    'Iris': datasets.load_iris(),
    'Digits': datasets.load_digits(),
    'Wine': datasets.load_wine(),
    'Breast Cancer': datasets.load_breast_cancer()
}

classifs = {
    "call": {
        "ZeroR": DummyClassifier(strategy="most_frequent", random_state=rand_state),
        "Aleatorio": DummyClassifier(strategy="uniform", random_state=rand_state),
        "Estratificado": DummyClassifier(strategy="stratified", random_state=rand_state),
        "OneR Probab": OneRProbabilistico(random_state=rand_state), 
        "Naive Bayes": GaussianNB(),
        "KMeans": Kcentroides(method=kmeans, random_state=rand_state),
        "KGA": Kcentroides(method=kga, random_state=rand_state),
        "KNN": KNeighborsClassifier(weights='uniform'), 
        "DistKNN": KNeighborsClassifier(weights='distance'),
        "Árvore Desc": DecisionTreeClassifier(random_state=rand_state),
        "Floresta": RandomForestClassifier(random_state=rand_state)
    },
    "p_grid": {
        "KMeans": {'estimator__k': [1,3,5,7]},
        "KGA": {'estimator__k': [1,3,5,7]},
        "KNN": {'estimator__n_neighbors': [1,3,5,7]}, 
        "DistKNN": {'estimator__n_neighbors': [1,3,5,7]},
        "Árvore Desc": {'estimator__max_depth': [None, 3, 5,10]},
        "Floresta": {'estimator__n_estimators': [10,20,50,100]}
    },
    "needs_discrete": ["OneR Probab"]
}

# %% [markdown]
# ### Inicializar dicionario que guardará os resultados

# %%
# Initializing results dict
results = {k: dict(dict.fromkeys(classifs["call"])) for k in  datasets_}

# %% [markdown]
# # RODANDO TREINO/TESTE DE AMBAS ETAPAS

# %%
# Processando
for dataset, ds_data in datasets_.items():
    for method, call in classifs["call"].items():
        if method not in classifs["p_grid"]:
            results[dataset][method] = phase_one(ds_data, call, method in classifs["needs_discrete"], rand_state)
        else:
            results[dataset][method] = phase_two(ds_data, call, classifs["p_grid"][method], rand_state)
        print(dataset, method, results[dataset][method]["status"], round(results[dataset][method]["time"], 2))

# %% [markdown]
# ### Salvar/Exibir tabelas de status (mean, std, lim inf, lim sup)
#     Comente as linhas especificadas caso não deseje salvar.

# %%
from pathlib import Path
import pandas as pd 
import copy

header = ["Média", "Desvio Padrão", "Lim. Inferior", "Lim. Superior"]
path = "tables/"

# Exibir e salvar tabelas de status no subdiretorio definido acima (latex e CSV)
for ds_name, ds_data in results.items():
    results_df = pd.DataFrame({method: v["status"] for method, v in ds_data.items()}).T
    print(results_df)

    # COMENTE AS LINHAS ABAIXO CASO NÃO DESEJE SALVAR
    Path(path).mkdir(parents=True, exist_ok=True)
    file_name = ds_name.replace(" ", "_")+'_status_table'
    with open(path + file_name + '.tex', 'w') as file:
        file.write(results_df.to_latex(float_format="%.3f", label=file_name, header=header))
    with open(path + '[CSV]'+ file_name + '.csv', 'w') as file:
        file.write(results_df.to_csv(float_format="%.3f", header=header))


# %% [markdown]
# ### Salvar/exibir matrizes de confusão dos algoritmos implementados
# %% [markdown]
#      Função para plotagem de matriz de confusão
# %%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict #Para heatmap
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
path = "plots/"
# Exibir e Salvar Matrizes de Confusão no diretorio definido acima
for ds_name in ["Iris", "Breast Cancer", "Wine"]:
    x, y = datasets_[ds_name].data, datasets_[ds_name].target
    labels = datasets_[ds_name].target_names

    for m in ["KMeans", "KGA", "OneR Probab"]:
        skf = StratifiedKFold(n_splits=10,random_state=rand_state, shuffle=True)

        classifier = classifs["call"][m]
        if hasattr(classifier, "k"):
            classifier.k = 7

        pipe_list = [('transformer', StandardScaler()), ('estimator', classifier)]
        if m == "OneR Probab": 
            discretizer = KBinsDiscretizer(2*len(np.unique(y)), encode='ordinal', strategy='kmeans')
            pipe_list.insert(1, ('discretizer', discretizer))
        pipeline = Pipeline(pipe_list)

        y_pred = cross_val_predict(pipeline, x, y, cv=skf)
        conf_mat = confusion_matrix(y, y_pred)

        plot_conf_mat(conf_mat, labels, ds_name + ' ' + m)

        filename = 'heatmap_'+ m + '_' + ds_name
        if m != "OneR Probab": filename += '_k7'

        # # Comente as duas linhas abaixo caso não deseje salvar no computador
        # Path(path).mkdir(parents=True, exist_ok=True)
        # plt.savefig(path + filename + '.png', dpi=200)

        plt.show()

# %% [markdown]
# ## BOXPLOTS DOS SCORES

# %%

path = "plots/"
# Exibir e Salvar Matrizes de Confusão no diretorio definido acima
for ds_name, ds_data in results.items():
    score_df = pd.DataFrame({method: v["scores"] for method, v in ds_data.items()}).T
    plot_dataset_boxplot(score_df, ds_name)
        
    # Comente as duas linhas abaixo caso não deseje salvar no computador
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path + "boxplot_" + ds_name + '_scores',dpi=200)

    plt.show()

# %% [markdown]
# ### EXIBIR/SALVAR TABELA DE TEMPOS

# %%
path = "tables/"

times = {}
for ds_name, ds_data in results.items():
    times[ds_name] = {method: v["time"] for method, v in ds_data.items()}

times_df = pd.DataFrame(times)

times_df.loc['Total por Dataset'] = times_df.sum()
times_df['Total por classif.'] = times_df.sum(axis=1)

print(times_df)

# Salvando a tabela de tempos na subpasta definida acima
Path(path).mkdir(parents=True, exist_ok=True)
with open(path+ "runtime_table.tex", 'w') as file:
    file.write(times_df.to_latex(float_format="%.2f", label="runtime_table"))
with open(path+ "[CSV]runtime_table.csv", 'w') as file:
    file.write(times_df.to_csv(float_format="%.2f"))

# %% [markdown]
# ## TABELA T-TEST e WILCOXON

# %%
from scipy.stats import ttest_rel, wilcoxon

# Gerando tabela do teste t e wilcoxon de cada dataset
p_value_dfs = {}
for ds_name, ds_data in results.items():
    score_df = pd.DataFrame({method: v["scores"] for method, v in ds_data.items()}).T
    score_values = score_df.values
    p_value_df = np.full((len(score_values),len(score_values)), 1.)

    # Rodando ttest e wilcoxon
    for i, r1 in enumerate(score_values):
        for j, r2 in enumerate(score_values):
            if (r1==r2).all(): continue
            p_value_df[i,j] = ttest_rel(r1, r2)[1] if j > i else wilcoxon(r1, r2)[1] 

    p_value_dfs[ds_name] = pd.DataFrame(p_value_df, columns=score_df.index, index=score_df.index)
    print(ds_name, p_value_dfs[ds_name])

# torna negrito resultados rejeitados (menores ou iguais que 0.05)
def bold_formatter(x):
    return "\\textbf{%.5f}" % x if x <= 0.05 else '%.5f' % x


# Exportando tabelas como CSV e Latex
path = "tables/"
for ds_name, ds_p_value in p_value_dfs.items():
    name = ds_name.replace(" ", "_")
    column_names = [x[:5] for x in ds_p_value.columns]

    # Aplicar o formatador para todas as colunas
    formatters = [bold_formatter]*len(ds_p_value.columns)

    # Salvar tabela latex e CSV no subdiretorio /tables
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(path + name + '_p_value_table.tex', 'w') as file:
        file.write(ds_p_value.to_latex(formatters=formatters, 
                                        escape=False, 
                                        index=False, 
                                        header=column_names, 
                                        label=name+"teste_pareado_table",
                                        caption= ds_name + ": p-values dos Testes Pareados. Teste t de Student na matrix triangular superior e Teste de Wilcoxon na matriz triangular inferior. Valores arredondados para cinco casas decimais."))

    with open(path + '[CSV]' + name + '_p_value_table.csv', 'w') as file:
        file.write(ds_p_value.to_csv(float_format="%.2e", header=column_names, index=column_names))


