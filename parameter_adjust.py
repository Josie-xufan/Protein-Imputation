start = time.process_time()
for j in range(0,correlation_saverx.shape[1]):  #correlation.shape[1]
    n_estimators = range(10, 501, 20)
    rmse_lightGBM = []
    rmse_GBDT = []
    rmse_randomforest = []
    rmse_xgboost = []
    r2_lightGBM = []
    r2_GBDT = []
    r2_randomforest = []
    r2_xgboost = []
    figure_rmse=[]
    figure_r2 = []
    for p in range(0,len(n_estimators)):
        param = {'n_estimators': n_estimators[p]}
        x_train,x_test,y_train,y_test,x_data,y_data= process(j,correlation_saverx,reap_mrna_saverx)
        x_train = x_train.values[:, 0:]
        x_test = x_test.values[:, 0:]
        y_train = y_train.values
        y_test = y_test.values
        rmse_lightGBM,r2_lightGBM,gbm=lightGBM_train_nocross(j,param,x_train, x_test, y_train, y_test)
        rmse_GBDT,r2_GBDT,gbd2=GBDT_train_nocross(j,param,x_train, x_test, y_train, y_test)
        rmse_randomforest, r2_randomforest, rf2 ,y_pred= randomforest_train_nocross(j, param, x_train, x_test, y_train, y_test)
        rmse_xgboost,r2_xgboost,xgb2=xgboost_train_nocross(j, param, x_train, x_test, y_train, y_test)

    rmse_lightGBM = DataFrame(rmse_lightGBM)
    rmse_GBDT = DataFrame(rmse_GBDT)
    rmse_randomforest = DataFrame(rmse_randomforest)
    rmse_xgboost = DataFrame(rmse_xgboost)


    figure_rmse = pd.concat([DataFrame(n_estimators), rmse_lightGBM, rmse_GBDT, rmse_randomforest, rmse_xgboost],axis=1)
    figure_rmse.columns = ['n_estimators', 'rmse_lightGBM', 'rmse_GBDT', 'rmse_randomforest', 'rmse_xgboost']
    plot_estimator_rmse(j,figure_rmse,figure_rmse.drop(['n_estimators'], axis=1).stack().min()-0.01,figure_rmse.drop(['n_estimators'], axis=1).stack().max()+0.01)

    r2_lightGBM = DataFrame(r2_lightGBM)
    r2_GBDT = DataFrame(r2_GBDT)
    r2_randomforest = DataFrame(r2_randomforest)
    r2_xgboost = DataFrame(r2_xgboost)

    figure_r2 = pd.concat([DataFrame(n_estimators),r2_lightGBM,r2_GBDT,r2_randomforest,r2_xgboost],axis=1)
    figure_r2.columns = ['n_estimators', 'r2_lightGBM', 'r2_GBDT', 'r2_randomforest', 'r2_xgboost']
    plot_estimator_r2(j, figure_r2, figure_r2.drop(['n_estimators'], axis=1).stack().min() - 0.01,figure_r2.drop(['n_estimators'], axis=1).stack().max() + 0.01)
for j in range(0,correlation_saverx.shape[1]):  # correlation.shape[1]
    max_depth = range(2, 21, 1)
    rmse_lightGBM = []
    rmse_GBDT = []
    rmse_randomforest = []
    rmse_xgboost = []
    r2_lightGBM = []
    r2_GBDT = []
    r2_randomforest = []
    r2_xgboost = []
    figure_rmse=[]
    figure_r2 = []
    for p in range(0, len(max_depth)):
        param = {'max_depth': max_depth[p]}
        x_train, x_test, y_train, y_test,x_data,y_data = process(j,correlation_saverx,reap_mrna_saverx)
        x_train = x_train.values[:, 0:]
        x_test = x_test.values[:, 0:]
        y_train = y_train.values
        y_test = y_test.values
        rmse_lightGBM, r2_lightGBM, gbm = lightGBM_train_nocross(j, param, x_train, x_test, y_train, y_test)
        rmse_GBDT, r2_GBDT, gbd2 = GBDT_train_nocross(j, param, x_train, x_test, y_train, y_test)
        rmse_randomforest, r2_randomforest, rf2 ,y_pred= randomforest_train_nocross(j, param, x_train, x_test, y_train, y_test)
        rmse_xgboost, r2_xgboost, xgb2 = xgboost_train_nocross(j, param, x_train, x_test, y_train, y_test)

    rmse_lightGBM = DataFrame(rmse_lightGBM)
    rmse_GBDT = DataFrame(rmse_GBDT)
    rmse_randomforest = DataFrame(rmse_randomforest)
    rmse_xgboost = DataFrame(rmse_xgboost)

    figure_rmse = pd.concat([DataFrame(max_depth), rmse_lightGBM, rmse_GBDT, rmse_randomforest, rmse_xgboost],axis=1)
    figure_rmse.columns = ['max_depth', 'rmse_lightGBM', 'rmse_GBDT', 'rmse_randomforest', 'rmse_xgboost']
    plot_maxdepth_rmse(j,figure_rmse,figure_rmse.drop(['max_depth'], axis=1).stack().min()-0.05,figure_rmse.drop(['max_depth'], axis=1).stack().max()+0.05)

    r2_lightGBM = DataFrame(r2_lightGBM)
    r2_GBDT = DataFrame(r2_GBDT)
    r2_randomforest = DataFrame(r2_randomforest)
    r2_xgboost = DataFrame(r2_xgboost)

    figure_r2 = pd.concat([DataFrame(max_depth), r2_lightGBM, r2_GBDT, r2_randomforest, r2_xgboost], axis=1)
    figure_r2.columns = ['max_depth', 'r2_lightGBM', 'r2_GBDT', 'r2_randomforest', 'r2_xgboost']
    plot_maxdepth_r2(j, figure_r2, figure_r2.drop(['max_depth'], axis=1).stack().min() - 0.01,figure_r2.drop(['max_depth'], axis=1).stack().max() + 0.01)
end = time.process_time()
print('start:',start)
print('end',end)
print('total:%.2fsecond'%(end-start))
