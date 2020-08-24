# Section 3.2(1) number of features is fixed as 2000, changing the samples' number

import pandas as pd
import numpy as np
from sklearn import model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score  #R square
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pylab as plt
import warnings


warnings.filterwarnings("ignore")

# Read RNA data (normalized)
cbmc_rna = pd.read_csv('../cmbc_rna_denoised.csv',header = 0, index_col=0)
cbmc_rna = pd.DataFrame(cbmc_rna)


print(cbmc_rna.shape)

# Read ADT data
cbmc_adt = pd.read_csv('../cmbc_adt_noPrefix.csv',header = 0, index_col = 0)
cbmc_adt = pd.DataFrame(cbmc_adt)

print(cbmc_adt.shape)

# Row is the sample
cbmc_rna_t = cbmc_rna.T
cbmc_adt_t = cbmc_adt.T

cor_matrix = pd.read_csv('../denoised_cor_matrix.csv', header = 0, index_col=0)
# cor_matrix = pd.read_csv('correlation_SAVERX.csv', header = 0, index_col=0)

# RNA as the row and ADT as the column
cor_matrix_t = cor_matrix.T

samples_num = [200,400,600,800,1200,1600,2000,2400,2800,3200,3600]

random_num = [1,13,17,58,96]

# repetition times for each samples number
repetition  = 5

fig_mean=pd.DataFrame(index=[s/2000 for s in samples_num],columns=['RF_test','MLP_test','RF_train','MLP_train'])
fig_std=pd.DataFrame(index=[s/2000 for s in samples_num],columns=['RF_test','MLP_test','RF_train','MLP_train'])
fig_mean[fig_mean!=0]=0
fig_std[fig_std!=0]=0

for j in range(len(samples_num)):

    print("Current samples_num is: ", samples_num[j])

    # std for all repetition RMSE
    rf_std = []
    rf_train_std = []
    mlp_std = []
    mlp_train_std = []

    for r in range(repetition):

        rf_RMSE = []
        rf_train_RMSE = []
        mlp_RMSE = []
        mlp_train_RMSE = []

        for i in range(10):   # 0~9

            print("Current ADT is: ", cor_matrix_t.columns[i])
            cur_adt = cor_matrix_t.columns[i]

            rna_sort = cor_matrix_t.sort_values(cur_adt, ascending=False)
            # Select top2000 PCC genes
            top20_rna_name = rna_sort[0:2000].index.tolist()

            # Combine RNA and ADT data
            mydata = []

            mydata = pd.concat([cbmc_rna_t[top20_rna_name], cbmc_adt_t[cur_adt]], axis=1)

            print('mydata.shape: ',mydata.shape)

            X = mydata.values[:,0:2000]
            y = mydata.values[:,2000]

            # divide training and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            print("X.shape: ", X.shape)

            # filter train_samples numbers
            X_train = pd.DataFrame(X_train)
            y_train = pd.DataFrame(y_train)
            X_train = X_train.sample(n=samples_num[j], random_state= random_num[r])
            y_train = y_train.sample(n=samples_num[j], random_state= random_num[r])

            print('RandomForest Start training...')
            # Ctreate and train model
            rf = RandomForestRegressor(n_estimators=200)

            rf.fit(X_train, y_train)
            rf_y_pred = rf.predict(X_test)
            rf_train_pred = rf.predict(X_train)

            rf_RMSE.append(mean_squared_error(y_test, rf_y_pred) ** 0.5)
            rf_train_RMSE.append(mean_squared_error(y_train, rf_train_pred) ** 0.5)

            #----------------------------------------------

            print('MLP Start training...')
            # Ctreate and train model
            mlp_reg = MLPRegressor(learning_rate_init=0.001, solver='sgd', learning_rate='adaptive', batch_size=32,
                                   validation_fraction=0.2, early_stopping=True, max_iter=1000, hidden_layer_sizes=(256, 64,),)         # 原先的learning_rate是0.01

            mlp_reg.fit(X_train, y_train)
            mlp_y_pred = mlp_reg.predict(X_test)
            mlp_train_pred = mlp_reg.predict(X_train)

            mlp_RMSE.append(mean_squared_error(y_test, mlp_y_pred) ** 0.5)
            mlp_train_RMSE.append(mean_squared_error(y_train, mlp_train_pred) ** 0.5)

        # After running all proteins, print ---------
        print("\n")
        print('RF total mean RMSE: ',np.mean(rf_RMSE))
        print('MLP total mean RMSE: ',np.mean(mlp_RMSE))

        print("\n")
        print('RF total train mean RMSE: ',np.mean(rf_train_RMSE))
        print('MLP total train mean RMSE: ',np.mean(mlp_train_RMSE))

        fig_mean.loc[samples_num[j] / 2000, "RF_test"] += np.mean(rf_RMSE)
        fig_mean.loc[samples_num[j] / 2000, "MLP_test"] += np.mean(mlp_RMSE)
        fig_mean.loc[samples_num[j] / 2000, "RF_train"] += np.mean(rf_train_RMSE)
        fig_mean.loc[samples_num[j] / 2000, "MLP_train"] += np.mean(mlp_train_RMSE)
        rf_std.append(np.mean(rf_RMSE))
        rf_train_std.append(np.mean(rf_train_RMSE))
        mlp_std.append(np.mean(mlp_RMSE))
        mlp_train_std.append(np.mean(mlp_train_RMSE))

    # After all repetitions...
    fig_mean.loc[samples_num[j] / 2000, "RF_test"] /= repetition
    fig_mean.loc[samples_num[j] / 2000, "MLP_test"] /= repetition
    fig_mean.loc[samples_num[j] / 2000, "RF_train"] /= repetition
    fig_mean.loc[samples_num[j] / 2000, "MLP_train"] /= repetition
    fig_std.loc[samples_num[j] / 2000, "RF_test"] = np.std(rf_std)
    fig_std.loc[samples_num[j] / 2000, "MLP_test"] = np.std(mlp_std)
    fig_std.loc[samples_num[j] / 2000, "RF_train"] = np.std(rf_train_std)
    fig_std.loc[samples_num[j] / 2000, "MLP_train"] = np.std(mlp_train_std)

# Write to Excel...
fig_mean.to_csv("fig_mean_samples_over_features.csv")
fig_std.to_csv("fig_std_samples_over_features.csv")

import matplotlib.pyplot as plt
n_genes = [0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]
data_1 = {
    'x': list(n_genes),
    'y': fig_mean["RF_test"],
    'yerr':fig_std["RF_test"]}
data_2 = {
    'x': list(n_genes),
    'y': fig_mean["RF_train"],
    'yerr':fig_std["RF_train"]}
data_3 = {
    'x': list(n_genes),
    'y': fig_mean["MLP_test"],
    'yerr':fig_std["MLP_test"]}
data_4 = {
    'x': list(n_genes),
    'y': fig_mean["MLP_train"],
    'yerr':fig_std["MLP_train"]}
# errorbar + fill_between
fig = plt.figure()
ax1 = fig.add_subplot(111)
for data in [data_1]: #RF_test
    line1 = ax1.errorbar(**data, alpha=.75,fmt=':',ecolor='#1f77b4',color='#1f77b4', capsize=3,capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25,facecolor = "#1f77b4")
for data in [data_2]: #RF_train
    line2 = ax1.errorbar(**data, alpha=.75,ecolor='#1f77b4',color='#1f77b4',capsize=3,capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25,facecolor = "#1f77b4")
for data in [data_3]: #MLP_test
    line3 = ax1.errorbar(**data, alpha=.75,fmt=':',ecolor='#ff7f0e',color='#ff7f0e', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25,facecolor = "#ff7f0e")
for data in [data_4]: #MLP_train
    line4 = ax1.errorbar(**data, alpha=.75,ecolor='#ff7f0e',color='#ff7f0e', capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
    plt.fill_between(**data, alpha=.25,facecolor = "#ff7f0e")
#plt.legend()
ax1.legend(handles=[line1,line2,line3,line4], labels=['RF(test)','RF(train)','MLP(test)','MLP(train)'], loc = (0.72,0.14) )
ax1.set_ylim(0.10,0.60)
ax1.set_title("RF and MLP performance")
ax1.set_ylabel('RMSE ')
ax1.set_xlabel("The ratio of number of samples over number of features")
ax1.set_xticks([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8])
fig.savefig('E:\\ML_py\\sample_split_CITE.pdf', dpi=600, format='pdf')

