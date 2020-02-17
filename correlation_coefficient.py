adt=reap_adt
pre_lightGBM_nosaverx=[]
pre_GBDT_nosaverx=[]
pre_random_nosaverx=[]
pre_xgboost_nosaverx=[]
pre_neuralnet_nosaverx=[]
pre_lightGBM_saverx=[]
pre_GBDT_saverx=[]
pre_random_saverx=[]
pre_xgboost_saverx=[]
pre_neuralnet_saverx=[]
param={}
pre_nosaverx=[]
pre_saverx=[]

for j in range(0,correlation_saverx.shape[1]):
    x_train_saverx, x_test_saverx, y_train_saverx, y_test, x_data_saverx, y_data_saverx = process(j,correlation_saverx,reap_mrna_saverx)
    x_train_nosaverx, x_test_nosaverx, y_train_nosaverx, y_test, x_data_nosaverx, y_data_nosaverx = process(j,correlation_nosaverx,reap_mrna_nosaverx)

    x_saverx = x_data_saverx.values[:, 0:]
    y_saverx = y_data_saverx.values
    x_nosaverx = x_data_nosaverx.values[:, 0:]
    y_nosaverx = y_data_nosaverx.values

    gbm = LGBMRegressor(**param, num_leaves=31, learning_rate=0.01, object='regression')
    y_pred_saverx = cross_valid_predict(gbm, x_saverx, y_saverx)
    y_pred_nosaverx = cross_valid_predict(gbm, x_nosaverx, y_nosaverx)
    y_pred_saverx = Series(y_pred_saverx)
    y_pred_nosaverx = Series(y_pred_nosaverx)
    y_pred_saverx.index = x_data_saverx.index
    y_pred_nosaverx.index = x_data_nosaverx.index
    pre_lightGBM_nosaverx.append(adt[adt.columns[j]].corr(y_pred_nosaverx))
    pre_lightGBM_saverx.append(adt[adt.columns[j]].corr(y_pred_saverx))

    gbd2 = ensemble.GradientBoostingRegressor(**param, learning_rate=0.01)
    y_pred_saverx = cross_valid_predict(gbd2, x_saverx, y_saverx)
    y_pred_nosaverx = cross_valid_predict(gbd2, x_nosaverx, y_nosaverx)
    y_pred_saverx = Series(y_pred_saverx)
    y_pred_nosaverx = Series(y_pred_nosaverx)
    y_pred_saverx.index = x_data_saverx.index
    y_pred_nosaverx.index = x_data_nosaverx.index
    pre_GBDT_nosaverx.append(adt[adt.columns[j]].corr(y_pred_nosaverx))
    pre_GBDT_saverx.append(adt[adt.columns[j]].corr(y_pred_saverx))

    rf2 = RandomForestRegressor(**param)
    y_pred_saverx = cross_valid_predict(rf2, x_saverx, y_saverx)
    y_pred_nosaverx = cross_valid_predict(rf2, x_nosaverx, y_nosaverx)
    y_pred_saverx = Series(y_pred_saverx)
    y_pred_nosaverx = Series(y_pred_nosaverx)
    y_pred_saverx.index = x_data_saverx.index
    y_pred_nosaverx.index = x_data_nosaverx.index
    pre_random_nosaverx.append(adt[adt.columns[j]].corr(y_pred_nosaverx))
    pre_random_saverx.append(adt[adt.columns[j]].corr(y_pred_saverx))

    xgb2 = xgb.XGBRegressor(**param, learning_rate=0.01, num_leaves=31)
    y_pred_saverx = cross_valid_predict(xgb2, x_saverx, y_saverx)
    y_pred_nosaverx = cross_valid_predict(xgb2, x_nosaverx, y_nosaverx)
    y_pred_saverx = Series(y_pred_saverx)
    y_pred_nosaverx = Series(y_pred_nosaverx)
    y_pred_saverx.index = x_data_saverx.index
    y_pred_nosaverx.index = x_data_nosaverx.index
    pre_xgboost_nosaverx.append(adt[adt.columns[j]].corr(y_pred_nosaverx))
    pre_xgboost_saverx.append(adt[adt.columns[j]].corr(y_pred_saverx))

    clf = MLPRegressor(**param, learning_rate_init=0.01, solver='sgd', learning_rate='adaptive', batch_size=32,validation_fraction=0.2, early_stopping=True, max_iter=1000, hidden_layer_sizes=(256, 64,))
    y_pred_saverx = cross_valid_predict(clf, x_saverx, y_saverx)
    y_pred_nosaverx = cross_valid_predict(clf, x_nosaverx, y_nosaverx)
    y_pred_saverx = Series(y_pred_saverx)
    y_pred_nosaverx = Series(y_pred_nosaverx)
    y_pred_saverx.index = x_data_saverx.index
    y_pred_nosaverx.index = x_data_nosaverx.index
    pre_neuralnet_nosaverx.append(adt[adt.columns[j]].corr(y_pred_nosaverx))
    pre_neuralnet_saverx.append(adt[adt.columns[j]].corr(y_pred_saverx))

pre_lightGBM_nosaverx=DataFrame(pre_lightGBM_nosaverx)
pre_GBDT_nosaverx=DataFrame(pre_GBDT_nosaverx)
pre_random_nosaverx=DataFrame(pre_random_nosaverx)
pre_xgboost_nosaverx=DataFrame(pre_xgboost_nosaverx)
pre_neuralnet_nosaverx=DataFrame(pre_neuralnet_nosaverx)

pre_lightGBM_saverx=DataFrame(pre_lightGBM_saverx)
pre_GBDT_saverx=DataFrame(pre_GBDT_saverx)
pre_random_saverx=DataFrame(pre_random_saverx)
pre_xgboost_saverx=DataFrame(pre_xgboost_saverx)
pre_neuralnet_saverx=DataFrame(pre_neuralnet_saverx)


pre_nosaverx = pd.concat([pre_lightGBM_nosaverx, pre_GBDT_nosaverx,pre_random_nosaverx,pre_xgboost_nosaverx,pre_neuralnet_nosaverx], axis=1)
pre_saverx = pd.concat([pre_lightGBM_saverx, pre_GBDT_saverx,pre_random_saverx,pre_xgboost_saverx,pre_neuralnet_saverx], axis=1)
DataFrame(pre_nosaverx).columns=['LightGBM','GBDT','RF','XGBoost','MLP']
DataFrame(pre_saverx).columns=['LightGBM','GBDT','RF','XGBoost','MLP']
pre_nosaverx.index=correlation_nosaverx.columns
pre_saverx.index=correlation_nosaverx.columns
pre_nosaverx.to_csv("E:\\REAP_cor_nosaverx.csv")
pre_saverx.to_csv("E:\\REAP_cor_saverx.csv")