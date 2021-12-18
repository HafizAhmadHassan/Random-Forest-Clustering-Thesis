import pandas as pd
import numpy as np
import os
import random

import random
import numpy as np
import pandas as pd
import uuid

import sklearn
from pandas.core.dtypes.common import is_string_dtype

from sklearn import preprocessing

from sklearn import *
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation

#from fastai.imports import *

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

#from data_Prep import *



from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


import numpy as np


class Nodes:
    def __init__(self,leafs,indleaf):
        self.leafs = leafs
        self.indleaf =indleaf

class SemiDf:
    def __init__(self,data,label):
        self.data = data
        self.label = label


class Treey:
    def __init__(self,index,nodes,weights):
        self.index = index
        self.nodes = nodes
        self.weights = weights



def proximityMatrix(model, X, normalize=True):
    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1 * np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat



def forestCaller(df_raw, label):
    """
    Case 3
    m = RandomForestClassifier(min_samples_leaf=10, max_depth=50, n_jobs=-1)
    m.fit(df_raw.sample(n=int(len(df_raw.index) * 0.8)), label)

    case 4
    m = RandomForestClassifier(min_samples_leaf=10, max_depth=50,max_features=0.5, n_jobs=-1)
    m.fit(df_raw.sample(n=int(len(df_raw.index) * 0.8)), label)

    case 5
    m = RandomForestClassifier(n_estimators=50,min_samples_leaf=10, max_depth=50, n_jobs=-1)
    m.fit(df_raw.sample(n=int(len(df_raw.index) * 0.8)), label)



    case 6
    m = RandomForestClassifier(n_estimators=50,min_samples_leaf=10, max_depth=50,max_features=0.5, n_jobs=-1)
    m.fit(df_raw.sample(n=int(len(df_raw.index) * 0.8)), label)




Analysis v1
    m = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, max_depth=50, max_features=0.5, n_jobs=-1)
    m.fit(df_raw, label)
    return m, m.score(df_raw, label)
Analysis v1
    m = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=50, max_features=0.5, n_jobs=-1)
    m.fit(df_raw, label)



v1 : Default parameters

v2 :
    m = RandomForestClassifier(min_samples_leaf=10, max_depth=50, n_jobs=-1)

v3 :
    m = RandomForestClassifier(min_samples_leaf=10, max_depth=50, max_features=0.5,n_jobs=-1)
v4:

    m = RandomForestClassifier(n_estimators=50,min_samples_leaf=10, max_depth=50, n_jobs=-1)

v5:
    m = RandomForestClassifier(n_estimators=50,min_samples_leaf=10, max_depth=50,max_features=0.5, n_jobs=-1)



    """

    m = RandomForestClassifier(n_jobs=-1)

    m.fit(df_raw.values, label)
    return m, m.score(df_raw, label)


def format_dataset(df_raw, labelname):
    df = df_raw.drop(labelname, axis=1)
    y = df_raw.labelname
    return df, y

def readAndSavefile(PATH,filename,ram=0):
    if filename in ["iris.csv", "glass-e.csv", "heart.csv", "WBC.csv", "wine.csv"]:
        df_raw = pd.read_csv(f'{PATH}{filename}', low_memory=False,sep=";")
        if (ram == 1):
            os.makedirs('tmp', exist_ok=True)
            df_raw.to_feather(f'tmp/{filename}-raw')
        return df_raw
    else:
        df_raw = pd.read_csv(f'{PATH}{filename}', low_memory=False, sep=",")
        if (ram == 1):
            os.makedirs('tmp', exist_ok=True)
            df_raw.to_feather(f'tmp/{filename}-raw')
        return df_raw


class Disti:
    def __init__(self, values, counting):
        self.values = values
        self.counting = counting


def synthetic_MarginalData(df_raw):
    columnsN = df_raw.columns
    list = []
    for colind in range(len(columnsN)):
        a = df_raw[columnsN[colind]].value_counts().axes
        b = df_raw[columnsN[colind]].value_counts().array
        list.append(Disti(a, b))
    arrRan = np.zeros(len(columnsN) - 1)

    columns = columnsN
    df = pd.DataFrame(columns=columns)

    for i in range(len(df_raw)):
        for colind in range(len(columnsN) - 1):
            arrRan[colind] = \
            random.choices(np.reshape(np.array(list[colind].values), (np.shape(list[colind].values)[1],)),
                           weights=np.array(list[colind].counting))[0]
        arr = np.append(arrRan, 0)
        df.loc[i] = (arr)
    return df


def makeDataset_UnSupervRF(df_raw, labelname):
    df_raw[[labelname]] = 1
    new_df = pd.concat([df_raw, synthetic_MarginalData(df_raw)])
    return new_df


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier





from IPython.display import display
import pandas as pd


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


################################################

def leaf_Weight_helper(RF,SmDt): #random forest and semisupervised data
    terminals=RF.apply(SmDt.data)
    nTrees = terminals.shape[1]
    leafs = []
    storeNodes= []
    for i in range(nTrees):
        uniqleaf = set(terminals[:,i])
        leafs.append(uniqleaf)
        mylist = []
        for j in uniqleaf:
            l=[index for index, value in enumerate(terminals[:,i]) if value == j]
            mylist.append(l)
        n = Nodes(uniqleaf,mylist)
        storeNodes.append(n)
        mylist=[]
    return storeNodes


def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()



def semi_supds(df,dr):
    df_ = []
    idx=0
    for i in df.y.unique():
        df_.append(df[df['y'] == i])
        remove_n = int(len(df_[idx])*dr)
        drop_indices = np.random.choice(df_[idx].index, remove_n, replace=False)
        df_[idx] = df_[idx].drop(drop_indices)
        idx = idx + 1
    df_row = pd.concat([df_[0], df_[1]])
    if len(df_) >= 2:
        for i in range(2,len(df_)):
            df_row= pd.concat([df_row, df_[i]])
    return df_row



import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
"""
def ()
    df = pd.read_excel('File.xlsx', sheetname='Sheet1')

    print("Column headings:")
    print(df.columns)
"""

def semiData_Gen(filepath,filename,labelname,dataRatio=0.7):
    semi_df=readAndSavefile(filepath, filename)
    num = np.random.choice(range(0, len(semi_df)),int(len(semi_df)*dataRatio),replace=False)
    if(type(semi_df[labelname][0]) is str):
        train_cats(semi_df)
        semi_df[labelname] = semi_df[labelname].cat.codes
    semi_df.at[num, labelname] = np.NAN
    lbl = semi_df[labelname]
    return semi_df.drop(labelname,axis=1),lbl

import math

def compute_entropy(ml, k):
    # calculate is nan
    # assume at max k+2 labels and
    arr = []
    for i in range(0, k + 2):
        arr.append(0)

    for i in range(0, len(ml)):
        if (math.isnan(ml[i])):
            arr[k + 1] = arr[k + 1] + 0
        else:
            arr[ml[i]] = arr[ml[i]] + 1

    s = np.sum(arr)
    arr = np.divide(arr, s)
    #print(arr)
    ent = 0
    for i in range(0, len(arr)):
        if(arr[i] != 0):
            ent = ent + arr[i] * np.log2(arr[i])
    ent = -1 * ent
    if(np.isnan(ent)):
        return 1
    return ent


def compute_weigth(ml, k):
    # calculate is nan
    # assume at max k+2 labels and
    arr = []
    for i in range(0, k + 2):
        arr.append(0)

    for i in range(0, len(ml)):
        if (math.isnan(ml[i])):
            arr[k + 1] = arr[k + 1] + 1
        else:
            arr[ml[i]] = arr[ml[i]] + 1
    #print(ml)
    #print(arr)

    """
    check if nan is present



    array_sum = np.sum(array)
    array_has_nan = np.isnan(array_sum)


    """
    ml2 = []
    array_sum = np.sum(ml)
    array_has_nan = np.isnan(array_sum)
    if (array_has_nan):
        # print(array_has_nan)
        for u in range(0, len(ml)):
            if (np.isnan(ml[u])):
                continue
            else:
                ml2.append(ml[u])
    # print("this is ml2",ml2)
    #ml = np.array(ml, dtype=float)
    s = np.sum(arr)

    a = arr[k + 1] / s

    b = ((s - arr[k + 1]) / s) * (1 + np.var(ml2))

    # b = (s - arr[k+1])/ s) * (1 + np.var(ml2)) good result
    # (s - arr[k+1])/s * (1 + np.var(ml2)) close to entropy
    if (np.isnan(a*b)):
        return 1
    return a * b

def compute_weigthee(ml, k):
    # calculate is nan
    # assume at max k+2 labels and
    arr = []
    for i in range(0, k + 2):
        arr.append(0)

    for i in range(0, len(ml)):
        if (math.isnan(ml[i])):
            arr[k + 1] = arr[k + 1] + 1
        else:
            arr[ml[i]] = arr[ml[i]] + 1
    #print(ml)
    #print(arr)

    """
    check if nan is present



    array_sum = np.sum(array)
    array_has_nan = np.isnan(array_sum)


    """
    ml2 = []
    array_sum = np.sum(ml)
    array_has_nan = np.isnan(array_sum)
    if (array_has_nan):
        # print(array_has_nan)
        for u in range(0, len(ml)):
            if (np.isnan(ml[u])):
                continue
            else:
                ml2.append(ml[u])
    # print("this is ml2",ml2)
    #ml = np.array(ml, dtype=float)
    s = np.sum(arr)

    a = arr[k + 1] / s

    b = ((s - arr[k + 1]) / s) * (1 + compute_entropy(ml2,k))

    # b = (s - arr[k+1])/ s) * (1 + np.var(ml2)) good result
    # (s - arr[k+1])/s * (1 + np.var(ml2)) close to entropy
    if (np.isnan(a*b)):
        return 1
    return a * b

def rescaling(arr):
    arr = np.array(arr)
    arr1 = np.sort(arr, axis=None)
    for i in range(0, len(arr1) - 1):
        s = arr1[i+1] - arr1[i]
        if (s > 1):
            arr1[i + 1] = arr1[i + 1] - (s - 1)
    return arr1

def proposal_3weigth(ml, k):
    x = compute_entropy(ml,k)
    x = 1 + x
    return 1/x

def Tree_leaf_weight(y,SmDt,k):
    mylist3 =[]
    len(y)
    weights= []
    listTree = []
    for j in range(len(y)):
        for i in y[j].indleaf:
            mylist3.append(SmDt.label[i])
            try:
            # print("yes")
                #t = compute_weigth(rescaling(i), k)
                t= proposal_3weigth(rescaling(i),k)
            except:
                t = 1
            # t = var(i)

            # if (math.isnan(t)):
            #    t = 1/k
            # if k-float(t) < 0:
            #   print(t," - ",k, " = ", k-t ,"  Hello")
            # print(t) k-(1-t) diverge
            # Exp 2 : k-t can have negative value
            # t-k good results but negative value
            # formula : frq(label1) + fequency(label2)/freq(all labels)
            # t/k experiment k+1/(t+1)
            # 1/(t+1) experiment best
            # 1/t+k
            weights.append(t)
        t=Treey(j, y[j] , weights)
        listTree.append(t)
        weights=[]
        mylist3=[]
    return listTree


def get_weighted_prm(tl,pmt):
    for ti in range(len(tl)):
        for i in range(len(tl[ti].nodes.indleaf)):
            if(len(tl[ti].nodes.indleaf[i])>1):
                for j in range(len(tl[ti].nodes.indleaf[i])):
                    for k in range(j+1,len(tl[ti].nodes.indleaf[i])):
                        inj= tl[ti].nodes.indleaf[i][j]
                        ink= tl[ti].nodes.indleaf[i][k]
                        pmt[inj][ink]=pmt[inj][ink]+ tl[ti].weights[i]
                        pmt[ink][inj]=pmt[ink][inj]+ tl[ti].weights[i]
            else:
                pmt[tl[ti].nodes.indleaf[i],tl[ti].nodes.indleaf[i]]= 0
    return pmt


##################### CLuster and Proximity

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import *
from sklearn_extra.cluster import KMedoids

from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt



def display_proximities(img , columns, rows):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    for i in range(0, columns * rows - 1):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img[i])
    plt.show()


def apply_cluster(dm,clus,pred_actual,n_ds):

    """
    This code cluster  data with different methods and save into file called storage

    clus: number of clustor

    f : filename to save

    dm : dissimalirity matrix of all datasets

    i: loop interation

    pred_actual : actual label

    n_ds : number of datasets

    """
    f = open("Storage.txt", "a")

    sfKmed = []
    cluster_c = []
    sfC = []
    cluster_p = []
    sfAp = []
    cluster_ap = []
    cluster_s = []
    sfSpec = []
    sfAlg = []
    cluster_alg = []
    sfS = []
    cluster_spec = []
    for i in range(0,n_ds):
        print(i)
        cluster_c.append(AgglomerativeClustering(n_clusters=clus[i],
                                                 affinity='precomputed',
                                                 linkage='complete'))
        cluster_c[i].fit_predict(dm[i])
        cluster_c[i].labels_ = cluster_c[i].labels_ + 1
        sfC.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_c[i].labels_))
        f.write(str(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_c[i].labels_)))
        f.write("\n")

        cluster_s.append(AgglomerativeClustering(n_clusters=clus[i],
                                                 affinity='precomputed',
                                                 linkage='single'))
        cluster_s[i].fit_predict(dm[i])
        cluster_s[i].labels_ = cluster_s[i].labels_ + 1
        sfS.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_s[i].labels_))

        f.write(str(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_s[i].labels_)))
        f.write("\n")

        cluster_spec.append(
            SpectralClustering(n_clusters=clus[i], affinity="rbf", assign_labels="discretize", random_state=0))
        cluster_spec[i].fit_predict(dm[i])
        cluster_spec[i].labels_ = cluster_spec[i].labels_ + 1
        pred_actual[i] = np.array(pred_actual[i])
        sfSpec.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_spec[i].labels_))

        f.write(str(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_spec[i].labels_)))
        f.write("\n")
        cluster_alg.append(AgglomerativeClustering(n_clusters=clus[i]))
        cluster_alg[i].fit_predict(dm[i])
        cluster_alg[i].labels_ = cluster_alg[i].labels_ + 1
        pred_actual[i] = np.array(pred_actual[i])

        sfAlg.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_alg[i].labels_))

        f.write(str(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_alg[i].labels_)))
        f.write("\n")
        cluster_ap.append(AffinityPropagation())

        cluster_ap[i].fit_predict(dm[i])

        cluster_ap[i].labels_ = cluster_ap[i].labels_ + 1
        pred_actual[i] = np.array(pred_actual[i])
        sfAp.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_ap[i].labels_))
        f.write(str(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_ap[i].labels_)))
        f.write("\n")

        cluster_p.append(KMedoids(n_clusters=clus[i], random_state=0))
        cluster_p[i].fit_predict(dm[i])
        cluster_p[i].labels_ = cluster_p[i].labels_ + 1
        pred_actual[i] = np.array(pred_actual[i])
        sfKmed.append((sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_p[i].labels_)))
        f.write(str(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_p[i].labels_)))
        f.write("\n")
        i = i + 1
        f.write("\n")


    display_proximities(dm, 3, 2)
    f.close()





