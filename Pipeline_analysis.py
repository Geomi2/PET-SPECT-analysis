from p_calc import pi_val
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LassoLarsCV

filename = '/workspace/yourfile'

#Create dataset
dataPET_rad=pd.read_excel(filename)

#Select target from your exce file
target="TARGET"
tar=dataPET_rad['TARGET']

p_v=[]
auc_p=[]
name=[]
pv=[]
ac=[]
nm=[]

#Models logistic
logisticRegr = LogisticRegression(n_jobs=-1)
#clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.005,max_depth=1)


#Calculate AUC,p val
for i in range(dataPET_rad.shape[1]-1):    
    X1 = dataPET_rad.iloc[:,i]    
    X_pet=pd.DataFrame(X1)                      
    y_pet=dataPET_rad.loc[:, target]    
    predictions=logisticRegr.fit(X_pet, y_pet)
    dtrain_predprob = predictions.predict_proba(X_pet)[:,1]  
    p,auc=pi_val(dtrain_predprob,y_pet)
    pv.append(p)
    ac.append(auc)
    nm.append(X1.name)
    #Remove features with p<0.05
    if p < 0.05:
         if auc < 0.5:
            p_v.append(p)
            auc_p.append(1-auc)
            name.append(X1.name)
         else:
            p_v.append(p)
            auc_p.append(auc)
            name.append(X1.name)
                
df1=pd.DataFrame(name,columns=["Feature"]) 
df2=pd.DataFrame(auc_p,columns=["AUC"])  
df3=pd.DataFrame(p_v,columns=["P values"])
df4=pd.DataFrame(nm,columns=["Feature"]) 
df5=pd.DataFrame(ac,columns=["TAUC"])  
df6=pd.DataFrame(pv,columns=["TP values"])
final_df=pd.concat((df1,df2,df3),axis=1)
final_df_total=pd.concat((df4,df5,df6),axis=1)


#Remove features with AUC < 0.6
final_df.drop(final_df[final_df['AUC'] <= 0.6].index,inplace=True)
#final_df.to_excel("output_pet_pl_auc_logistic.xlsx",index=False)
to_drop1=list(final_df.iloc[:,0])
dataPET_rad1 = dataPET_rad.drop(to_drop1, axis=1)
dataPET_rad = dataPET_rad.drop(dataPET_rad1.columns, axis=1)

#Remove features with r > 0.85
cor_matrix = dataPET_rad.corr('spearman').abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]

#Costruct final dataset
final_set = dataPET_rad.drop(to_drop, axis=1)
final_set=pd.concat((final_set,tar),axis=1)
#final_set.to_excel("output_pet_pl_auc_logistic_filtered.xlsx",index=False)

#Initialize Sequential feature selection
sfs = sfs(logisticRegr,k_features=(1,15),  forward=True, verbose=2, scoring='roc_auc',cv=5,n_jobs=-1)

#Apply min-max scaling
scaler = MinMaxScaler()
XX = final_set.loc[:,final_set.columns!=target]
XX_pet=scaler.fit_transform(XX)
XX_pet=pd.DataFrame(XX_pet,columns=XX.columns)
yy_pet=final_set.loc[:, target]

#Fit the model
sfs1 = sfs.fit(XX_pet,yy_pet)

#Save the best feature combination parameters
feat_names = list(sfs1.k_feature_names_)
#feat_names_lasso = list(lasso.feature_names_in_)
print(feat_names)
new_data = XX[feat_names]
new_data=pd.concat((new_data,tar),axis=1)


#Saved excel file for further process in MedCalc Software as described in paper
save_path = '/your_save_path'
new_data.to_excel(save_path,index=False)
