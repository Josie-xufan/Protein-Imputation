# Protein Imputation
## prediction_function.py
The functions used in the further analysis are prepared here. And the data including XXX_mrna_nosaverx, XXX_mrna_saverx, XXX_adt, correlation_nosaverx and correlation_saverx is imported.
1. The CITE-seq and REAP-seq data should be downloaded from the NCBI GEO database (https://www.ncbi.nlm.nih.gov/geo/). Each dataset contains the information of the the transcriptional abundance for RNAs and the expression levels for cell-surface proteins. Please follow the preprocessing methods mentioned in the protocols of the CITE-seq (https://www.nature.com/articles/nmeth.4380) and REAP-seq (https://www.nature.com/articles/nbt.3973). After preprocessing, you should follow the SAVER-X installation pipline (https://github.com/jingshuw/SAVERX) to denoise the RNAs counts.So you've got the RNA data before and after the de-noising with the processing,named as XXX_mrna_nosaverx and XXX_mrna_saverx in the code separately.And the preprocessed surface protein data is prepared,named as XXX_adt. XXX can be reap or cite for the different datasets.
1. Pearson correlation coefficient between proteins and RNAs before and after de-noising can be calculated based on the prepared data, named as correlation_nosaverx and correlation_saverx. 

## ctpnet.py
You can follow the cTPnet installation pipline (https://github.com/zhouzilu/cTPnet) to get the prediction of surface proteins before and after de-noising. And then calculate the performance evaluations of the cTP-net.

# supplementary
## supplementary files
Here are the supplementary files mentioned in our manuscript.

## supplementary trial
### overfitting analysis
To test whether overfitting is a major reason of the observed relatively poor performance of neural networks in our study, we performed additional experiments on the real datasets. Specifically, we split the CITE-seq dataset into different portions as the training and testing sets, changing the ratio of number of samples over the number of features. The performance was evaluated using the RMSE on both training and test datasets. The number of features in each training set was fixed to 2000, whereas the numbers of samples ranged from 100 to 3600. The input is the denoised RNA data and CLR-transformed protein expression data of CITE-seq-CBMC.
### feature selection analysis
To check if additional RNAs could be incorporated in the random forest model, we designed an experiment to incorporate 10 to 100 RNAs (in the increment of 10) in the model. The input is the denoised RNA data and CLR-transformed protein expression data of CITE-seq-CBMC.
### DTs
In order to further analyzing the component decision tree (DTs) of RF to gain more insight about its behavior, we calculated the feature importance of three different DTs randomly chosen from all DTs that were used to generate the RF model.



 
