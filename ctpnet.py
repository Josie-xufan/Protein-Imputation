y_test = pd.read_excel('E:\\ctpnet_CITE\\reap_adt.xlsx',sheet_name=[1],header=0,index_col=0)
y_test = DataFrame(y_test[1])
y_test = pd.DataFrame(y_test.values.T, index=y_test.columns, columns=y_test.index)

y_pred_nosaverx = pd.read_excel('E:\\ctpnet_CITE\\y_pred_ctpnet_nosaverx.xlsx',sheet_name=[1],header=0,index_col=0)
y_pred_nosaverx = DataFrame(y_pred_nosaverx[1])
y_pred_nosaverx = pd.DataFrame(y_pred_nosaverx.values.T, index=y_pred_nosaverx.columns, columns=y_pred_nosaverx.index)

y_pred_saverx = pd.read_excel('E:\\ctpnet_CITE\\y_pred_ctpnet_saverx.xlsx',sheet_name=[1],header=0,index_col=0)
y_pred_saverx = DataFrame(y_pred_saverx[1])
y_pred_saverx = pd.DataFrame(y_pred_saverx.values.T, index=y_pred_saverx.columns, columns=y_pred_saverx.index)

rmse_ctpnet=[]
r2_ctpnet=[]
pre_ctpnet=[]
for j in range(0,len(y_test.columns)):
    rmse_ctpnet.append(np.sqrt(mean_squared_error(y_pred_saverx[y_pred_saverx.columns[j]], y_test[y_test.columns[j]])))
    r2_ctpnet.append(r2_score(y_pred_saverx[y_pred_saverx.columns[j]], y_test[y_test.columns[j]]))
    pre_ctpnet.append(y_test[y_test.columns[j]].corr(y_pred_saverx[y_pred_saverx.columns[j]]))

rmse_ctpnet=DataFrame(rmse_ctpnet)
rmse_ctpnet.index=y_test.columns
r2_ctpnet=DataFrame(r2_ctpnet)
r2_ctpnet.index=y_test.columns

rmse_ctpnet.to_csv("E:\\ctpnet_CITE\\rmse_ctpnet_saverx.csv")
r2_ctpnet.to_csv("E:\\ctpnet_CITE\\r2_ctpnet_saverx.csv")
DataFrame(pre_ctpnet).to_csv('E:/ctpnet_CITE/cor_saverx.csv')


