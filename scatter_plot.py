param={}
rna_name=['HLA-DRA','CD27','CD4','CD69','CD28','CD19','CD14'] #For REAP-seq data
adt_name=[3,12,13,20,21,26,42]
adt_name_ctpnet=[0,3,4,6,7,8,11]

mrna_saverx=reap_mrna_saverx
mrna_saverx = DataFrame(mrna_saverx)

mrna_nosaverx=reap_mrna_nosaverx
mrna_nosaverx = DataFrame(mrna_nosaverx)

y_test_ctpnet = pd.read_excel('E:\\ctpnet_CITE\\reap_adt.xlsx',sheet_name=[1],header=0,index_col=0)
y_test_ctpnet = DataFrame(y_test_ctpnet[1])
y_test_ctpnet = pd.DataFrame(y_test_ctpnet.values.T, index=y_test_ctpnet.columns, columns=y_test_ctpnet.index)

y_pred_ctpnet_nosaverx = pd.read_excel('E:\\ctpnet_CITE\\y_pred_ctpnet_nosaverx.xlsx', sheet_name=[1], header=0,index_col=0)
y_pred_ctpnet_nosaverx = DataFrame(y_pred_ctpnet_nosaverx[1])
y_pred_ctpnet_nosaverx = pd.DataFrame(y_pred_ctpnet_nosaverx.values.T, index=y_pred_ctpnet_nosaverx.columns,columns=y_pred_ctpnet_nosaverx.index)

y_pred_ctpnet_saverx = pd.read_excel('E:\\ctpnet_CITE\\y_pred_ctpnet_saverx.xlsx', sheet_name=[1], header=0,index_col=0)
y_pred_ctpnet_saverx = DataFrame(y_pred_ctpnet_saverx[1])
y_pred_ctpnet_saverx = pd.DataFrame(y_pred_ctpnet_saverx.values.T, index=y_pred_ctpnet_saverx.columns,columns=y_pred_ctpnet_saverx.index)

adt=reap_adt

for j in range(0,len(rna_name)):
    x_train_saverx, x_test_saverx, y_train_saverx, y_test, x_data_saverx, y_data_saverx = process(adt_name[j], correlation_saverx,reap_mrna_saverx)
    x_train_nosaverx, x_test_nosaverx, y_train_nosaverx, y_test, x_data_nosaverx, y_data_nosaverx = process(adt_name[j], correlation_nosaverx,reap_mrna_nosaverx)

    x_saverx = x_data_saverx.values[:, 0:]
    y_saverx = y_data_saverx.values
    x_nosaverx = x_data_nosaverx.values[:, 0:]
    y_nosaverx = y_data_nosaverx.values

    rf2 = RandomForestRegressor(**param)
    y_pred_saverx = cross_valid_predict(rf2, x_saverx, y_saverx)
    y_pred_nosaverx = cross_valid_predict(rf2, x_nosaverx, y_nosaverx)

    y_pred_saverx = Series(y_pred_saverx)
    y_pred_nosaverx = Series(y_pred_nosaverx)
    y_pred_saverx.index = x_data_saverx.index
    y_pred_nosaverx.index = x_data_nosaverx.index

    f=plt.figure(figsize=(5, 5))
    plt.xlabel("RNA " + str(rna_name[j]))  # x轴上的名字
    plt.ylabel("Protein " + str(rna_name[j]))  # y轴上的名字
    plt.title('Cor: ' + str(round(adt[adt.columns[adt_name[j]]].corr(mrna_nosaverx[rna_name[j]]), 2)))
    plt.scatter(DataFrame(mrna_nosaverx[rna_name[j]]), DataFrame(adt[adt.columns[adt_name[j]]]), alpha=0.6)
    f.savefig('E:\\scatter_plot\\'+str(j)+'one.svg',dpi=600,format='svg')
    plt.clf()

    plt.figure(figsize=(5, 5))
    f, ax = plt.subplots(figsize=(5, 5))
    plt.xlabel("cTPnet predicted " + str(rna_name[j]))
    plt.ylabel("Protein " + str(rna_name[j]))
    plt.title('Cor: ' + str(round(adt[adt.columns[adt_name[j]]].corr(y_pred_ctpnet_nosaverx[y_pred_ctpnet_nosaverx.columns[adt_name_ctpnet[j]]]),2)))
    plt.scatter(DataFrame(y_pred_ctpnet_nosaverx[y_pred_ctpnet_nosaverx.columns[adt_name_ctpnet[j]]]),DataFrame(adt[adt.columns[adt_name[j]]]), alpha=0.6)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    f.savefig('E:\\scatter_plot\\' + str(j) + 'two.svg', dpi=600, format='svg')
    plt.clf()

    plt.figure(figsize=(5, 5))
    f, ax = plt.subplots(figsize=(5, 5))
    plt.xlabel("RF predicted " + str(rna_name[j]))
    plt.ylabel("Protein " + str(rna_name[j]))
    plt.title('Cor: ' + str(round(adt[adt.columns[adt_name[j]]].corr(y_pred_nosaverx), 2)))
    plt.scatter(DataFrame(y_pred_nosaverx), DataFrame(adt[adt.columns[adt_name[j]]]), alpha=0.4)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    f.savefig('E:\\scatter_plot\\' + str(j) + 'three.svg', dpi=600, format='svg')
    plt.clf()

    f=plt.figure(figsize=(5, 5))
    plt.xlabel("Denoised RNA " + str(rna_name[j]))  # x轴上的名字
    plt.ylabel("Protein " + str(rna_name[j]))  # y轴上的名字
    plt.title('Cor: ' + str(round(adt[adt.columns[adt_name[j]]].corr(mrna_saverx[rna_name[j]]), 2)))
    plt.scatter(DataFrame(mrna_saverx[rna_name[j]]), DataFrame(adt[adt.columns[adt_name[j]]]), alpha=0.6)
    f.savefig('E:\\scatter_plot\\' + str(j) + 'four.svg', dpi=600, format='svg')
    plt.clf()

    plt.figure(figsize=(5, 5))
    f, ax = plt.subplots(figsize=(5, 5))
    plt.xlabel("Denoised cTPnet predicted " + str(rna_name[j]))
    plt.ylabel("Protein " + str(rna_name[j]))
    plt.title('Cor: ' + str(round(adt[adt.columns[adt_name[j]]].corr(y_pred_ctpnet_saverx[y_pred_ctpnet_saverx.columns[adt_name_ctpnet[j]]]),2)))
    plt.scatter(DataFrame(y_pred_ctpnet_saverx[y_pred_ctpnet_saverx.columns[adt_name_ctpnet[j]]]),DataFrame(adt[adt.columns[adt_name[j]]]), alpha=0.6)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    f.savefig('E:\\scatter_plot\\' + str(j) + 'five.svg', dpi=600, format='svg')
    plt.clf()

    plt.figure(figsize=(5, 5))
    f, ax = plt.subplots(figsize=(5, 5))
    plt.xlabel("Denoised RF predicted " + str(rna_name[j]))
    plt.ylabel("Protein " + str(rna_name[j]))
    plt.title('Cor: ' + str(round(adt[adt.columns[adt_name[j]]].corr(y_pred_saverx), 2)))
    plt.scatter(DataFrame(y_pred_saverx), DataFrame(adt[adt.columns[adt_name[j]]]), alpha=0.4)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    f.savefig('E:\\scatter_plot\\' + str(j) + 'six.svg', dpi=600, format='svg')
    plt.clf()
    plt.close('all')



