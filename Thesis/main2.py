# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

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

#from Thesis.fastai.imports import *

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn_extra.cluster import KMedoids

#from scipy.cluster.hierarchy import dendrogram, linkage
#from scipy.spatial.distance import squareform

from Thesis.data_Prep import *
#from Thesis.fastai.tabular import *

#!
#!arr = ["BreastTissues.csv", "auto-mpg.csv", "Lung.csv", "Parinkson.csv", "iris.csv", "glass-e.csv", "heart.csv",
#!       "WBC.csv", "wine.csv"]


#####---new----



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation




def display_proximities(img , columns, rows):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    for i in range(0, columns * rows - 1):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img[i])
    plt.show()




import os



##################



        #if line.find(",") != -1 :
        #    print(line.split(","))


# Press the green button in the gutter to run the script.
from pandas.plotting import table
from openpyxl import Workbook

def createplot(desc,i,k):
    #create a subplot without frame
    plot = plt.subplot(111, frame_on=False)

    #remove axis
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)

    #create the table plot and position it in the upper left corner
    table(plot, desc,loc='upper right')

    #save the plot as a png file
    plt.savefig(f'desc_plot{k}{i}.png')

def avg_cal(path,si,ei):
    df2 = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=list('ABCDEF'))
    for j in range(0, 20):
        f = open(f'{path}/storage{j}', "r")

        line = f.readlines()

        for j in range(si, ei):
            line[j] = line[j].replace('\n', '')
            line[j] = line[j].split(",")
            x = np.array(line[j])
            y = x.astype(np.float)
            df = pd.DataFrame([y],columns=list('ABCDEF'))
            df2 = df2.append(df, ignore_index=True)
    return df2

if __name__ == '__main__':

    semi_df = []
    lbl = []
    SmDt = []
    y = []
    tl = []
    pmt = []
    pmt2 = []

    s_cluster_c = []
    s_sfC = []
    s_cluster_s = []

    s_sfS = []
    s_cluster_spec = []
    s_sfSpec = []

    s_cluster_alg = []
    s_sfAlg = []
    s_cluster_ap = []
    s_sfAp = []

    s_cluster_p = []

    s_sfKmed = []
    df_raw=[]

    pred_actual=[]
    c_df =[]
    RF=[]
    sRF=[]
    Prm=[]
    dm =[]
    cluster_c=[]
    sfC=[]
    i=0
    arr = "Lung.csv"

    df_raw.append(readAndSavefile("datasets/", arr))
    pred_actual.append(list(df_raw[i].y))

    c_df.append(semi_supds(df_raw[i], 0.3))

    rf, srf = forestCaller(c_df[i].drop("y", axis=1), c_df[i].y)

    RF.append(rf)
    sRF.append(srf)

    Prm.append(proximityMatrix(RF[i], df_raw[i].drop("y", axis=1), normalize=True))
    dm.append(1 - Prm[i])
    dm[i] = dm[i][:int(np.shape(dm[i])[0]), :int(np.shape(dm[i])[0])]
    dm[i] = preprocessing.normalize(dm[i])

    print(dm[i])
    try:
        cluster_c.append(AgglomerativeClustering(n_clusters=clus[i],
                                                 affinity='precomputed',
                                                 linkage='complete'))
        cluster_c[i].fit_predict(dm[i])
        cluster_c[i].labels_ = cluster_c[i].labels_ + 1
        sfC.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_c[i].labels_))
        print(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_c[i].labels_),4)))
        print(",")
    except:
        print("0.0")
        print(",")



"""
    for j in range(0, 20):
        unsuperv_rf_proximity('PyCharm', j, "0.3")


    for j in range(0, 20):
        unsuperv_rf_proximity('PyCharm', j, "0.5")



    for j in range(0, 20):
        unsuperv_rf_proximity('PyCharm', j, "0.7")



    for j in range(0, 20):
        unsuperv_rf_proximity('PyCharm', j, "0.9")


    """



   ## Dataset 0 dataframe


"""
    data = { 
        'Height': [5.1, 6.2, 5.1, 5.2] 
      } 
   
   
        # Convert the dictionary into DataFrame 
            df = pd.DataFrame(data) 
  
        # Using 'Address' as the column name and equating it to the list 
        df2 = df.assign(address = ['Delhi', 'Bangalore', 'Chennai', 'Patna']) 
   
        # Observe the result 
            df2 
            
"""

            #createplot(df.describe(),j,k)


"""
   df=avg_cal("0.5", 6, 7)
   display_all(df.describe(include='all').T)

   df=avg_cal("0.5", 9, 10)
   display_all(df.describe(include='all').T)

   df = avg_cal("0.5", 10, 11)
   display_all(df.describe(include='all').T)

   df=avg_cal("0.5", 13, 14)
   display_all(df.describe(include='all').T)

   df=avg_cal("0.5", 14, 15)
   display_all(df.describe(include='all').T)

   df=avg_cal("0.5", 17, 18)
   display_all(df.describe(include='all').T)

   df=avg_cal("0.5", 18, 19)
   display_all(df.describe(include='all').T)


"""

   #calculate_avg()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
