param={}
adt_name=[3,12,13,20,21,26,42]
for j in range(0,len(adt_name)):
    x_train_saverx, x_test_saverx, y_train_saverx, y_test, x_data_saverx, y_data_saverx = process(adt_name[j],correlation_saverx,reap_mrna_saverx)

    x_tr = x_train_saverx.values[:, 0:]
    x_te = x_test_saverx.values[:, 0:]
    y_tr = y_train_saverx.values
    y_te = y_test.values

    gbm = LGBMRegressor(**param, num_leaves=31, learning_rate=0.01, object='regression')
    gbm.fit(x_tr, y_tr)
    lightGBM_feature(j, gbm, x_train_saverx, correlation_saverx)

    gbd2 = ensemble.GradientBoostingRegressor(**param, learning_rate=0.01)
    gbd2.fit(x_tr, y_tr)
    GBDT_feature(j, gbd2, x_train_saverx, correlation_saverx)

    rf2 = RandomForestRegressor(**param)
    rf2.fit(x_tr, y_tr)
    randomforest_feature(j, rf2, x_train_saverx, correlation_saverx)

    xgb2 = xgb.XGBRegressor(**param, learning_rate=0.01, num_leaves=31)
    xgb2.fit(x_tr, y_tr)
    xgboost_feature(j, xgb2, x_train_saverx, correlation_saverx)
