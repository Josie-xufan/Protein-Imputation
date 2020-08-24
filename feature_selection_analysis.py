#problem 1 gene_split
import pandas as pd
from scipy.stats import pearsonr
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt

rna_log=pd.read_csv("E:\\ML_py\\data\\rna_denoised_CITE(original).csv",header=0,index_col=[0],sep=',')
rna_log = pd.DataFrame(rna_log.values.T, index=rna_log.columns, columns=rna_log.index)
adt_clr=pd.read_csv("E:\\ML_py\\data\\adt_clr_CITE.csv",header=0,index_col=[0],sep=',')
adt_clr =pd.DataFrame(adt_clr.values.T, index=adt_clr.columns, columns=adt_clr.index)
param={'n_estimators':200}
rf = RandomForestRegressor(**param)
MLP = MLPRegressor(learning_rate_init=0.001, solver='sgd', learning_rate='adaptive', batch_size=32,validation_fraction=0.2, early_stopping=True, max_iter=1000, hidden_layer_sizes=(256, 64,))

repetition=5
n_genes=range(10,110,10)
fig_mean=pd.DataFrame(index=n_genes,columns=['rmse','r2','cor'])
fig_std=pd.DataFrame(index=n_genes,columns=['rmse','r2','cor'])
fig_mean[fig_mean!=0]=0
fig_std[fig_std!=0]=0
for c in range(0,len(n_genes)):
    std_rmse = []
    std_r2 = []
    std_cor = []
    for r in range(0,repetition):
        index_rmse = []
        index_r2 = []
        index_cor = []
        for i in range(0, adt_clr.shape[1]):
            cor = []
            for j in range(0, rna_log.shape[1]):
                cor.append(abs(pearsonr(adt_clr[adt_clr.columns[i]], rna_log[rna_log.columns[j]])[0]))

            cor = DataFrame(cor)
            cor.index = rna_log.columns
            cor.columns = ['Pearson']
            cor = cor.sort_values(axis=0, ascending=False, by='Pearson')
            cor = cor[0:n_genes[c]]

            rna_change = rna_log[cor.index]
            x_train, x_test, y_train, y_test = train_test_split(rna_change, adt_clr[adt_clr.columns[i]], test_size=0.3)
            y_pred = MLP.fit(x_train.values, y_train.values).predict(x_test.values)
            index_rmse.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            index_r2.append(metrics.r2_score(y_test, y_pred))
            index_cor.append(pearsonr(y_pred, y_test)[0])

        fig_mean.loc[n_genes[c], "rmse"] += np.mean(index_rmse)
        fig_mean.loc[n_genes[c], "r2"] += np.mean(index_r2)
        fig_mean.loc[n_genes[c], "cor"] += np.mean(index_cor)
        std_rmse.append(np.mean(index_rmse))
        std_r2.append(np.mean(index_r2))
        std_cor.append(np.mean(index_cor))



    fig_mean.loc[n_genes[c], "rmse"] /= repetition
    fig_std.loc[n_genes[c], "rmse"] = np.std(std_rmse)
    fig_mean.loc[n_genes[c], "r2"] /= repetition
    fig_std.loc[n_genes[c], "r2"] = np.std(std_r2)
    fig_mean.loc[n_genes[c], "cor"] /= repetition
    fig_std.loc[n_genes[c], "cor"] = np.std(std_cor)

fig_mean.to_csv("E:\\ML_py\\data\\fig_mean_mlp_repetition_100_REAP.csv")
fig_std.to_csv("E:\\ML_py\\data\\fig_std_mlp_repetition_100_REAP.csv")

import matplotlib.pyplot as plt

data_1 = {
    'x': list(n_genes),
    'y': fig_mean["rmse"],
    'yerr': fig_std["rmse"]}
data_2 = {
    'x': list(n_genes),
    'y': fig_mean["r2"],
    'yerr': fig_std["r2"]}
data_3 = {
    'x': list(n_genes),
    'y': fig_mean["cor"],
    'yerr': fig_std["cor"]}
# errorbar + fill_between
fig = plt.figure()
ax1 = fig.add_subplot(111)
for data in [data_1]:
    line1 = ax1.errorbar(**data, alpha=.75, fmt='g:', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25, facecolor="green")
ax2 = ax1.twinx()  # this is the important function
for data in [data_2]:
    line2 = ax2.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25)

for data in [data_3]:
    line3 = ax2.errorbar(**data, alpha=.75, fmt=':', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25)

plt.legend(handles=[line1, line2, line3], labels=['RMSE', 'R\u00b2', 'Cor'], loc='center right')

# ax1.grid()
# ax1.set_ylim(0,2)
# ax2.set_ylim(0.88,0.925)
ax1.set_title("RF performance")
ax1.set_ylabel('RMSE')
ax1.set_xlabel("The number of feature RNAs")
ax2.set_ylabel('Cor/R\u00b2')
ax1.set_xticks(n_genes)
fig.savefig('E:\\ML_py\\gene_split_rf_repetition_100_CITE.pdf', dpi=600, format='pdf')