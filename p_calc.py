import numpy as np
import scipy.stats as st



def auc_roc(X, Y):    
    return 1/(len(X)*len(Y)) * sum([kernel(x, y) for x in X for y in Y])

def kernel(X, Y):
    return .5 if Y==X else int(Y < X)

def structural_components(X, Y):
    V10 = [1/len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1/len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01
    
def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])

def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    if auc_A!=auc_B:
        res1=(auc_A - auc_B)/((var_A + var_B - 2*covar_AB)**(.5))
    else:
        res1=0
    return res1
    
  

# Model A (random) vs. "good" model B
#preds_A =np.linspace(.5, .5, len(y))
def group_preds_by_label(preds, actual):
    X = [p for (p, a) in zip(preds, actual) if a]
    Y = [p for (p, a) in zip(preds, actual) if not a]
    return X, Y

def pi_val(preds_B,y):
    preds_A =np.linspace(.5, .5, len(preds_B))
    X_A, Y_A = group_preds_by_label(preds_A,y)
    V_A10, V_A01 = structural_components(X_A, Y_A)
    auc_A = auc_roc(X_A, Y_A)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
             + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
    
    X_B, Y_B = group_preds_by_label(preds_B, y)
   
    V_B10, V_B01 = structural_components(X_B, Y_B)
    
    auc_B = auc_roc(X_B, Y_B)
    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
             + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
    # Two tailed test
    z = -z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = st.norm.sf(abs(z))*2   
    #print("AUC: {0:.3f}, p-value: {1:.3f}".format(auc_B,p))
    #print("p-value: {0:.3f}".format(p))
    return p,auc_B


def pi_val_models( preds_A,y1,preds_B,y2):
    #preds_A =np.linspace(.5, .5, len(preds_B))
    X_A, Y_A = group_preds_by_label(preds_A,y1)
    V_A10, V_A01 = structural_components(X_A, Y_A)
    auc_A = auc_roc(X_A, Y_A)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
             + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
    
    X_B, Y_B = group_preds_by_label(preds_B, y2)
   
    V_B10, V_B01 = structural_components(X_B, Y_B)
    
    auc_B = auc_roc(X_B, Y_B)
    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
             + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
    # Two tailed test
    z = -z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = st.norm.sf(abs(z))*2
   
    # print('\nP-value difference')
    # print("Stenosis + Plaque AUC: {0:.3f}, Stenosis + Plaque + Radiomics AUC: {1:.3f}, p-value: {2:.3f}".format(auc_A,auc_B,p))
    return p,auc_A,auc_B



