import pandas as pd
from pandas import Series
from pandas.core.frame import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
plt.switch_backend('TkAgg')
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import warnings
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import math
import seaborn as sns
from sklearn.model_selection import cross_validate
from PIL import Image
import os
import random
from sklearn.model_selection import cross_val_predict

reap_mrna_saverx=pd.read_csv("E:\\ctpnet_CITE\\reap_mrna_saverx.csv",header=0,index_col=[0],sep=',')
reap_mrna_saverx = DataFrame(reap_mrna_saverx)
reap_mrna_saverx = pd.DataFrame(reap_mrna_saverx.values.T, index=reap_mrna_saverx.columns, columns=reap_mrna_saverx.index)
reap_mrna_saverx.to_pickle('reap_mrna_saverx')

reap_mrna_nosaverx=pd.read_csv("E:\\ctpnet_CITE\\reap_mrna_nosaverx.csv",header=0,index_col=[0],sep=',')
reap_mrna_nosaverx = DataFrame(reap_mrna_nosaverx)
reap_mrna_nosaverx = pd.DataFrame(reap_mrna_nosaverx.values.T, index=reap_mrna_nosaverx.columns, columns=reap_mrna_nosaverx.index)
reap_mrna_nosaverx.to_pickle('reap_mrna_nosaverx')

reap_adt=pd.read_csv("E:\\ctpnet_CITE\\reap_adt.csv",header=0,index_col=[0],sep=',')
reap_adt = DataFrame(reap_adt)
reap_adt = pd.DataFrame(reap_adt.values.T, index=reap_adt.columns, columns=reap_adt.index)
reap_adt.to_pickle('reap_adt')

correlation_nosaverx=pd.read_csv("E:\\ctpnet_CITE\\correlation_nosaverx.csv",header=0,index_col=[0],sep=',')
correlation_nosaverx.to_pickle('correlation_nosaverx')
correlation_saverx=pd.read_csv("E:\\ctpnet_CITE\\correlation_saverx.csv",header=0,index_col=[0],sep=',')
correlation_saverx.to_pickle('correlation_saverx')


start = time.process_time()
def train_test_data(nadt,mydata,correlation):
    adt_predict = correlation.columns[nadt]
    cols = [i for i in mydata.columns if i not in [adt_predict]]
    x = mydata[cols]
    y = mydata[adt_predict]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train,x_test,y_train,y_test
def print_best_score(gsearch, param_test):
    # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return best_parameters
def stdDev(test,predict):
    tot = 0.0
    for i in range(0,len(test)-1):
        tot += (test[i]-predict[0][i]) ** 2
    return (tot / len(test)) ** 0.5  # Square root of mean difference
def CV(test,predict):
    mean = sum(test)/float(len(test))
    try:
        return stdDev(test,predict)/mean
    except ZeroDivisionError:
        return float('nan')
def MAPE(test,predict):
    tot=0.0
    for i in range(0,len(test)-1):
        tot += (abs((test[i]-predict[0][i])/test[i]))
    return (tot/len(test))
def  MAD(test,predict):
    tot=0.0
    for i in range(0,len(test)-1):
        tot += abs(test[i]-predict[0][i])
    return (tot/len(test))
def muti_score(model):
    warnings.filterwarnings('ignore')
    mean_squared_error = cross_val_score(model, x_data, y_data, scoring='neg_mean_squared_error', cv=10)
    r2 = cross_val_score(model, x_data, y_data, scoring='r2', cv=10)
    return mean_squared_error,r2
def cross_valid(model,x_data,y_data):
    scores = cross_validate(model, x_data, y_data, scoring=['neg_mean_squared_error','r2','neg_mean_absolute_error'],cv = 10, return_train_score = False)
    return scores
def cross_valid_predict(model,x_data,y_data):
    prediction=cross_val_predict(model, x_data,y_data, cv=10)
    return prediction

def process(j,correlation,reap_mrna_data):
    mydata=[]
    rna_sort = correlation.sort_values(correlation.columns[j], ascending=False)
    high_corr_mrna = rna_sort[0:20].index.tolist()
    mydata = pd.concat([reap_adt[correlation.columns[j]], reap_mrna_data[high_corr_mrna]], axis=1)
    x_train, x_test, y_train, y_test = train_test_data(j,mydata,correlation)
    x_data = mydata.drop(correlation.columns[j], axis=1)
    y_data = mydata[correlation.columns[j]]
    return x_train,x_test,y_train,y_test,x_data,y_data
def lightGBM_train(j,param,x,y):
    gbm = LGBMRegressor(**param,num_leaves=31,learning_rate=0.01,object='regression')
    scores = cross_valid(gbm,x,y)
    cross_val_MSE=scores['test_neg_mean_squared_error']
    cross_val_r2=scores['test_r2']
    cross_val_MAE=scores['test_neg_mean_absolute_error']
    with open('lightGBM'+str(j)+'.pkl',"wb") as f:
    pickle.dump(gbm, f)
    rmse_lightGBM.append(np.sqrt(abs(cross_val_MSE.mean())))
    r2_lightGBM.append(cross_val_r2.mean())
    mae_lightGBM.append(cross_val_MAE.mean())
    return rmse_lightGBM,r2_lightGBM,mae_lightGBM,gbm
def lightGBM_feature(j,gbm,x_train,correlation):
    importances_lightGBM=[]
    headers = ["name", "score"]
    values = sorted(zip(x_train.columns, gbm.feature_importances_), key=lambda x: x[1] * -1)
    importances_lightGBM=pd.DataFrame(values)
    importances_lightGBM.columns=headers
    importances_lightGBM.to_csv('E:/feature importance/table/'+str(j)+'lightGBM.csv', index=False)
    importances_lightGBM.set_index('name')
    plt.figure(figsize=(35, 25))
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.tick_params(labelsize=30)
    plt.barh(importances_lightGBM['name'], importances_lightGBM['score'],  tick_label=importances_lightGBM['name'])
    plt.xlabel('LightGBM importances',fontsize=35)
    plt.ylabel('Features',fontsize=35)
    plt.title(correlation.columns[j],fontsize=35)
    plt.savefig('E:/feature importance/figure/'+str(j)+'lightGBM.svg',dpi=600,format='svg')
    plt.clf()
    plt.close('all')
def GBDT_train(j,param,x,y):
    gbd2 = ensemble.GradientBoostingRegressor(**param,learning_rate=0.01)
    scores = cross_valid(gbd2,x,y)
    cross_val_MSE=scores['test_neg_mean_squared_error']
    cross_val_r2=scores['test_r2']
    cross_val_MAE=scores['test_neg_mean_absolute_error']
    with open('GBDT'+str(j)+'.pkl',"wb") as f:
    pickle.dump(gbd2, f)
    rmse_GBDT.append(np.sqrt(abs(cross_val_MSE.mean())))
    r2_GBDT.append(cross_val_r2.mean())
    mae_GBDT.append(cross_val_MAE.mean())
    return rmse_GBDT,r2_GBDT,mae_GBDT,gbd2
def GBDT_feature(j,gbd2,x_train,correlation):
    importances_GBDT = []
    headers = ["name", "score"]
    values = sorted(zip(x_train.columns, gbd2.feature_importances_), key=lambda x: x[1] * -1)
    importances_GBDT=pd.DataFrame(values)
    importances_GBDT.columns=headers
    importances_GBDT.to_csv('E:/feature importance/table/'+str(j)+'GBDT.csv', index=False)
    importances_GBDT.set_index('name')
    plt.figure(figsize=(35, 25))
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.tick_params(labelsize=30)
    plt.barh(importances_GBDT['name'], importances_GBDT['score'], tick_label=importances_GBDT['name'])
    plt.xlabel('GBDT importances',fontsize=35)
    plt.ylabel('Features',fontsize=35)
    plt.title(correlation.columns[j],fontsize=35)
    plt.savefig('E:/feature importance/figure/'+str(j)+'GBDT.svg',dpi=600,format='svg')
    plt.clf()
    plt.close('all')
def randomforest_train(j,param,x,y):
    rf2 = RandomForestRegressor(**param)
    scores = cross_valid(rf2,x,y)
    cross_val_MSE=scores['test_neg_mean_squared_error']
    cross_val_r2=scores['test_r2']
    cross_val_MAE=scores['test_neg_mean_absolute_error']
    with open('randomforest'+str(j)+'.pkl',"wb") as f:
    pickle.dump(rf2, f)
    rmse_randomforest.append(np.sqrt(abs(cross_val_MSE.mean())))
    r2_randomforest.append(cross_val_r2.mean())
    mae_randomforest.append(cross_val_MAE.mean())
    return rmse_randomforest,r2_randomforest,mae_randomforest,rf2
def randomforest_feature(j,rf2,x_train,correlation):
    importances_randomforest=[]
    headers = ["name", "score"]
    values = sorted(zip(x_train.columns, rf2.feature_importances_), key=lambda x: x[1] * -1)
    importances_randomforest=pd.DataFrame(values)
    importances_randomforest.columns=headers
    importances_randomforest.to_csv('E:/feature importance/table/'+str(j)+'random.csv', index=False)
    importances_randomforest.set_index('name')
    plt.figure(figsize=(35, 25))
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.tick_params(labelsize=30)
    plt.barh(importances_randomforest['name'], importances_randomforest['score'],tick_label=importances_randomforest['name'])
    plt.xlabel('RF importances',fontsize=35)
    plt.ylabel('Features',fontsize=35)
    plt.title(correlation.columns[j],fontsize=35)
    plt.savefig('E:/feature importance/figure/'+str(j)+'random.svg',dpi=600,format='svg')
    plt.clf()
    plt.close('all')
def xgboost_train(j,param,x,y):
    xgb2 = xgb.XGBRegressor(**param,learning_rate=0.01,num_leaves=31)
    scores = cross_valid(xgb2, x, y)
    cross_val_MSE=scores['test_neg_mean_squared_error']
    cross_val_r2=scores['test_r2']
    cross_val_MAE=scores['test_neg_mean_absolute_error']
    with open('xgboost'+str(j)+'.pkl',"wb") as f:
    pickle.dump(xgb2, f)
    rmse_xgboost.append(np.sqrt(abs(cross_val_MSE.mean())))
    r2_xgboost.append(cross_val_r2.mean())
    mae_xgboost.append(cross_val_MAE.mean())
    return rmse_xgboost,r2_xgboost,mae_xgboost,xgb2
def xgboost_feature(j,xgb2,x_train,correlation):
    importances_xgboost=[]
    headers = ["name", "score"]
    values = sorted(zip(x_train.columns, xgb2.feature_importances_), key=lambda x: x[1] * -1)
    importances_xgboost=pd.DataFrame(values)
    importances_xgboost.columns=headers
    importances_xgboost.to_csv('E:/feature importance/table/'+str(j)+'xgboost.csv', index=False)
    importances_xgboost.set_index('name')
    plt.figure(figsize=(35, 25))
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.tick_params(labelsize=30)
    plt.barh(importances_xgboost['name'], importances_xgboost['score'], tick_label=importances_xgboost['name'])
    plt.xlabel('XGBoost importances',fontsize=35)
    plt.ylabel('Features',fontsize=35)
    plt.title(correlation.columns[j],fontsize=35)
    plt.savefig('E:/feature importance/figure/'+str(j)+'xgboost.svg',dpi=600,format='svg')
    plt.clf()
    plt.close('all')
def neuralnet_train(j,x,y):
    clf = MLPRegressor(learning_rate_init=0.01, solver='sgd', learning_rate='adaptive', batch_size=32,validation_fraction=0.2, early_stopping=True, max_iter=1000, hidden_layer_sizes=(256, 64,))
    scores = cross_valid(clf, x, y)
    cross_val_MSE=scores['test_neg_mean_squared_error']
    cross_val_r2=scores['test_r2']
    cross_val_MAE=scores['test_neg_mean_absolute_error']
    with open('neuralnet' + str(j) + '.pkl', "wb") as f:
    pickle.dump(clf, f)
    rmse_neuralnet.append(np.sqrt(abs(cross_val_MSE.mean())))
    r2_neuralnet.append(cross_val_r2.mean())
    mae_neuralnet.append(cross_val_MAE.mean())
    return rmse_neuralnet,r2_neuralnet,mae_neuralnet,clf

def lightGBM_train_nocross(j,param,x_train, x_test, y_train, y_test):
    gbm = LGBMRegressor(**param,num_leaves=31,learning_rate=0.01,object='regression')
    gbm.fit(x_train, y_train)
    y_pred = gbm.predict(x_test)
    y_pred = DataFrame(y_pred)
    rmse_lightGBM.append(np.sqrt(mean_squared_error(y_pred, y_test)))
    r2_lightGBM.append(r2_score(y_test, y_pred))
    return rmse_lightGBM,r2_lightGBM,gbm
def GBDT_train_nocross (j,param,x_train, x_test, y_train, y_test):
    gbd2 = ensemble.GradientBoostingRegressor(**param,learning_rate=0.01)
    gbd2.fit(x_train, y_train)
    y_pred = gbd2.predict(x_test)
    y_pred = DataFrame(y_pred)
    rmse_GBDT.append(np.sqrt(mean_squared_error(y_pred, y_test)))
    r2_GBDT.append(r2_score(y_test, y_pred))
    return rmse_GBDT,r2_GBDT,gbd2
def randomforest_train_nocross(j,param,x_train, x_test, y_train, y_test):
    rf2 = RandomForestRegressor(**param)
    rf2.fit(x_train, y_train)
    y_pred = rf2.predict(x_test)
    y_pred = DataFrame(y_pred)
    rmse_randomforest.append(np.sqrt(mean_squared_error(y_pred, y_test)))
    r2_randomforest.append(r2_score(y_test, y_pred))
    return rmse_randomforest,r2_randomforest,rf2,y_pred
def xgboost_train_nocross(j,param,x_train, x_test, y_train, y_test):
    xgb2 = xgb.XGBRegressor(**param,learning_rate=0.01,num_leaves=31)
    xgb2.fit(x_train, y_train)
    y_pred = xgb2.predict(x_test)
    y_pred = DataFrame(y_pred)
    rmse_xgboost.append(np.sqrt(mean_squared_error(y_pred, y_test)))
    r2_xgboost.append(r2_score(y_test, y_pred))
    return rmse_xgboost,r2_xgboost,xgb2
def neuralnet_train_nocross(j,param,x_train, x_test, y_train, y_test):
    clf = MLPRegressor(**param,learning_rate_init=0.01, solver='sgd', learning_rate='adaptive', batch_size=32,validation_fraction=0.2, early_stopping=True, max_iter=1000, hidden_layer_sizes=(256, 64,))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred = DataFrame(y_pred)
    rmse_neuralnet.append(np.sqrt(mean_squared_error(y_pred, y_test)))
    r2_neuralnet.append(r2_score(y_test, y_pred))
    return rmse_neuralnet,r2_neuralnet,clf

def plot_estimator_rmse(j,figure_rmse,min,max):
    plt.figure()
    plt.xlabel("n_estimators")
    plt.ylabel("RMSE")
    plt.title(correlation_saverx.columns[j])
    plt.scatter(figure_rmse['n_estimators'], figure_rmse['rmse_randomforest'], alpha=0.6, label="RF")
    plt.scatter(figure_rmse['n_estimators'], figure_rmse['rmse_lightGBM'], alpha=0.6, label="LightGBM")
    plt.scatter(figure_rmse['n_estimators'], figure_rmse['rmse_GBDT'], alpha=0.6, label="GBDT")
    plt.scatter(figure_rmse['n_estimators'], figure_rmse['rmse_xgboost'], alpha=0.6, label="XGBoost")
    plt.legend()
    plt.savefig('E:/robustness/estimator/rmse/' + str(j) + 'estimators.svg',dpi=600,format='svg')
    plt.clf()
def plot_estimator_r2(j,figure_r2,min,max):
    plt.figure()
    plt.xlabel("n_estimators")
    plt.ylabel("R\u00b2")
    plt.title(correlation_saverx.columns[j])
    plt.scatter(figure_r2['n_estimators'], figure_r2['r2_randomforest'], alpha=0.6, label="RF")
    plt.scatter(figure_r2['n_estimators'], figure_r2['r2_lightGBM'], alpha=0.6, label="LightGBM")
    plt.scatter(figure_r2['n_estimators'], figure_r2['r2_GBDT'], alpha=0.6, label="GBDT")
    plt.scatter(figure_r2['n_estimators'], figure_r2['r2_xgboost'], alpha=0.6, label="XGBoost")
    plt.legend()
    plt.savefig('E:/robustness/estimator/r2/' + str(j) + 'estimators.svg',dpi=600,format='svg')
    plt.clf()
def plot_maxdepth_rmse(j,figure_rmse,min,max):
    plt.figure()
    plt.xlabel("max_depth")
    plt.ylabel("RMSE")
    plt.title(correlation_saverx.columns[j])
    plt.scatter(figure_rmse['max_depth'], figure_rmse['rmse_randomforest'], alpha=0.6, label="RF")
    plt.scatter(figure_rmse['max_depth'], figure_rmse['rmse_lightGBM'], alpha=0.6, label="LightGBM")
    plt.scatter(figure_rmse['max_depth'], figure_rmse['rmse_GBDT'], alpha=0.6, label="GBDT")
    plt.scatter(figure_rmse['max_depth'], figure_rmse['rmse_xgboost'], alpha=0.6, label="XGBoost")
    plt.legend()
    plt.savefig('E:/robustness/maxdepth/rmse/' + str(j) + 'max_depth.svg',dpi=600,format='svg')
    plt.clf()
def plot_maxdepth_r2(j,figure_r2,min,max):
    plt.figure()
    plt.xlabel("max_depth")
    plt.ylabel("R\u00b2")
    plt.title(correlation_saverx.columns[j])
    plt.scatter(figure_r2['max_depth'], figure_r2['r2_randomforest'], alpha=0.6, label="RF")
    plt.scatter(figure_r2['max_depth'], figure_r2['r2_lightGBM'], alpha=0.6, label="LightGBM")
    plt.scatter(figure_r2['max_depth'], figure_r2['r2_GBDT'], alpha=0.6, label="GBDT")
    plt.scatter(figure_r2['max_depth'], figure_r2['r2_xgboost'], alpha=0.6, label="XGBoost")
    plt.legend()
    plt.savefig('E:/robustness/maxdepth/r2/' + str(j) + 'max_depth.svg',dpi=600,format='svg')
    plt.clf()
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)
end = time.process_time()
print('start:',start)
print('end',end)
print('total:%.2fsecond'%(end-start))










