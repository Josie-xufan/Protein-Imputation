from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
rna_log=pd.read_csv("E:\\ML_py\\data\\rna_denoised_CITE(original).csv",header=0,index_col=[0],sep=',')
rna_log = pd.DataFrame(rna_log.values.T, index=rna_log.columns, columns=rna_log.index)
adt_clr=pd.read_csv("E:\\ML_py\\data\\adt_clr_CITE.csv",header=0,index_col=[0],sep=',')
adt_clr =pd.DataFrame(adt_clr.values.T, index=adt_clr.columns, columns=adt_clr.index)
param={'n_estimators':200,'max_depth':10}
random=range(0,3)

# all protein
f,ax=plt.subplots(10,3,figsize=(30, 50)) #figsize 30宽，50高
for i in range(0,adt_clr.shape[1]):
    for c in range(0,len(random)):
        regressor = DecisionTreeRegressor(**param, random_state=random[c])
        cor=[]
        for j in range(0,rna_log.shape[1]):
            cor.append(pearsonr(adt_clr[adt_clr.columns[i]], rna_log[rna_log.columns[j]])[0])

        cor = DataFrame(cor)
        cor.index = rna_log.columns
        cor.columns = ['Pearson']
        cor = cor.sort_values(axis=0, ascending=False, by='Pearson')
        cor = cor[0:20] #the top 20 genes

        rna_change = rna_log[cor.index]
        x_train, x_test, y_train, y_test = train_test_split(rna_change, adt_clr[adt_clr.columns[i]], test_size=0.3,random_state=0)
        dtree = regressor.fit(x_train.values, y_train.values)

        importance=dtree.feature_importances_
        importance = DataFrame(importance)
        importance.index=rna_change.columns
        importance.columns=['importance']
        importance = importance.sort_values(axis=0, ascending=False, by='importance')

        ax[i][c].tick_params()
        ax[i][c].barh(importance.index, importance['importance'], tick_label=importance.index)
        ax[i][c].set_xlabel('Decision tree importances')
        ax[i][c].set_ylabel('Features')
        ax[i][c].set_title(adt_clr.columns[i]+" random_state: "+str(random[c]))
f.tight_layout()
f.savefig('E:\\ML_py\\decision_tree_feature_importance.png', dpi=600, format='png')
f.clf()

# protein CD19
i=8
rf = RandomForestRegressor(**param)
cor=[]
for j in range(0,rna_log.shape[1]):
    cor.append(pearsonr(adt_clr[adt_clr.columns[i]], rna_log[rna_log.columns[j]])[0])

cor = DataFrame(cor)
cor.index = rna_log.columns
cor.columns = ['Pearson']
cor = cor.sort_values(axis=0, ascending=False, by='Pearson')
cor = cor[0:20] #the top 20 genes
rna_change = rna_log[cor.index]
x_train, x_test, y_train, y_test = train_test_split(rna_change, adt_clr[adt_clr.columns[i]], test_size=0.3,random_state=0)
rf = rf.fit(x_train.values, y_train.values)

f,ax=plt.subplots(2,2,figsize=(70, 50)) #figsize (figsize[2]*35,figsize[1]*25)
importance = rf.feature_importances_
importance = DataFrame(importance)
importance.index = rna_change.columns
importance.columns = ['importance']
importance = importance.sort_values(axis=0, ascending=False, by='importance')
ax[0][0].tick_params(labelsize=40)
ax[0][0].barh(importance.index, importance['importance'], tick_label=importance.index)
ax[0][0].set_xlabel('\n RF importances \n', fontsize=45)
ax[0][0].set_ylabel('Features', fontsize=45)
ax[0][0].set_title(adt_clr.columns[i] + " \n", fontsize=45)
for j in range(1,4):
    importance = rf.estimators_[j-1].feature_importances_
    importance = DataFrame(importance)
    importance.index = rna_change.columns
    importance.columns = ['importance']
    importance = importance.sort_values(axis=0, ascending=False, by='importance')
    ax[j//2][j%2].tick_params(labelsize=40)
    ax[j//2][j%2].barh(importance.index, importance['importance'], tick_label=importance.index)
    ax[j//2][j%2].set_xlabel('\n DT importances', fontsize=45)
    ax[j//2][j%2].set_ylabel('Features', fontsize=45)
    ax[j//2][j%2].set_title(adt_clr.columns[i] + " estimators_["+str(j-1)+"]\n", fontsize=45)
f.tight_layout()
f.savefig('E:\\ML_py\\decision_tree_feature_importance_CD19.png', dpi=300, format='png')
f.clf()