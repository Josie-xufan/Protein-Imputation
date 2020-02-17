start = time.process_time()
rmse=[]
r_score=[]
mae=[]

rmse_lightGBM = []
rmse_GBDT = []
rmse_randomforest = []
rmse_xgboost = []
rmse_neuralnet=[]

r2_lightGBM = []
r2_GBDT = []
r2_randomforest = []
r2_xgboost = []
r2_neuralnet=[]

mae_lightGBM = []
mae_GBDT = []
mae_randomforest = []
mae_xgboost = []
mae_neuralnet=[]
param={}
for j in range(0,correlation_nosaverx.shape[1]):
    x_train, x_test, y_train, y_test ,x_data,y_data= process(j,correlation_nosaverx,reap_mrna_nosaverx)
    x = x_data.values[:, 0:]
    y=  y_data.values
    rmse_lightGBM,r2_lightGBM,mae_lightGBM,gbm=lightGBM_train(j,param,x,y)
    rmse_GBDT,r2_GBDT,mae_GBDT,gbd2=GBDT_train(j,param,x,y)
    rmse_randomforest,r2_randomforest,mae_randomforest,rf2=randomforest_train(j,param,x,y)
    rmse_xgboost,r2_xgboost,mae_xgboost,xgb2=xgboost_train(j, param, x, y)
    rmse_neuralnet,r2_neuralnet,mae_neuralnet,clf=neuralnet_train(j,x,y)

rmse_lightGBM = DataFrame(rmse_lightGBM)
rmse_GBDT = DataFrame(rmse_GBDT)
rmse_randomforest = DataFrame(rmse_randomforest)
rmse_xgboost = DataFrame(rmse_xgboost)
rmse_neuralnet=DataFrame(rmse_neuralnet)

r2_lightGBM = DataFrame(r2_lightGBM)
r2_GBDT = DataFrame(r2_GBDT)
r2_randomforest = DataFrame(r2_randomforest)
r2_xgboost = DataFrame(r2_xgboost)
r2_neuralnet=DataFrame(r2_neuralnet)

mae_lightGBM = DataFrame(mae_lightGBM)
mae_GBDT = DataFrame(mae_GBDT)
mae_randomforest = DataFrame(mae_randomforest)
mae_xgboost = DataFrame(mae_xgboost)
mae_neuralnet=DataFrame(mae_neuralnet)

rmse = pd.concat([rmse_lightGBM, rmse_GBDT,rmse_randomforest,rmse_xgboost,rmse_neuralnet], axis=1)
rmse.columns=['lightGBM','GBDT','randomforest','xgboost','neuralnet']

r_score = pd.concat([r2_lightGBM, r2_GBDT,r2_randomforest,r2_xgboost,r2_neuralnet], axis=1)
r_score.columns=['lightGBM','GBDT','randomforest','xgboost','neuralnet']

mae = pd.concat([mae_lightGBM, mae_GBDT,mae_randomforest,mae_xgboost,mae_neuralnet], axis=1)
mae.columns=['lightGBM','GBDT','randomforest','xgboost','neuralnet']

rmse.index = correlation_saverx.columns
r_score.index=correlation_saverx.columns
mae.index = correlation_saverx.columns

rmse.to_csv("E:\\REAP_rmse_nosaverx_estimator200.csv")
r_score.to_csv("E:\\REAP_r2_nosaverx_estimator200.csv")
mae.to_csv("E:\\REAP_mae_nosaverx.csv")

end = time.process_time()
print('start:',start)
print('end',end)
print('total:%.2f second'%(end-start))
