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

# from Thesis.fastai.imports import *

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn_extra.cluster import KMedoids

# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.spatial.distance import squareform

from Thesis.data_Prep import *
# from Thesis.fastai.tabular import *

# !
# !arr = ["BreastTissues.csv", "auto-mpg.csv", "Lung.csv", "Parinkson.csv", "iris.csv", "glass-e.csv", "heart.csv",
# !       "WBC.csv", "wine.csv"]


#####---new----


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation


def intit():
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


def display_proximities(img, columns, rows):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    for i in range(0, columns * rows - 1):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img[i])
    plt.show()


import os


##################


# if line.find(",") != -1 :
#    print(line.split(","))


def unsuperv_rf_proximity(name, j, dir):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    arr = ["BreastTissues.csv", "auto-mpg.csv", "Lung.csv", "Parinkson.csv", "iris.csv", "glass-e.csv", "heart.csv",
           "WBC.csv", "wine.csv"]

    # arr = ["iris.csv"]

    path2 = f'{dir}'
    if (os.path.isdir(path2) == False):
        os.mkdir(f'{dir}')
    f = open(f'{dir}/storage{j}', "a")

    clus = [6, 2, 3, 2, 3, 4, 2, 2, 3]
    index = 0

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

    df_raw = []
    pred_actual = []
    c_df = []
    RF = []
    sRF = []
    i = 0
    Prm = []
    dm = []
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

    """
    semi Supervised Arguements

    dr: data ratios

    semi_df: datafarame semisupervised


    lbl : label 

    SmDt : semisupervised data objects without label

    y: is used to store each dataset store in each random
    forest store the index of leaf and leafs


    e.g y[0].leafs y[1].indleaf

    tl : tree leaf (for each leag store weight)

    Pmt: proximity matrix

    pmt2: proximity weighted matrix 
    """

    # dr = [0.3, 0.5, 0.7, 0.9]

    for val in arr:

        f.write(f'dataset {i}\n')

        df = readAndSavefile("datasets/", val)
        # print(semi_supds(df, 0.5))

        df_raw.append(readAndSavefile("datasets/", val))
        pred_actual.append(list(df_raw[i].y))

        c_df.append(makeDataset_UnSupervRF(df_raw[i], "y"))

        rf, srf = forestCaller(c_df[i].drop("y", axis=1), c_df[i].y)

        RF.append(rf)
        sRF.append(srf)

        Prm.append(proximityMatrix(RF[i], c_df[i].drop("y", axis=1), normalize=True))
        dm.append(1 - Prm[i])
        dm[i] = dm[i][:int(np.shape(dm[i])[0] / 2), :int(np.shape(dm[i])[0] / 2)]
        dm[i] = preprocessing.normalize(dm[i])
        try:
            cluster_c.append(AgglomerativeClustering(n_clusters=clus[i],
                                                     affinity='precomputed',
                                                     linkage='complete'))
            cluster_c[i].fit_predict(dm[i])
            cluster_c[i].labels_ = cluster_c[i].labels_ + 1
            sfC.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_c[i].labels_))
            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_c[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue

        try:
            cluster_s.append(AgglomerativeClustering(n_clusters=clus[i],
                                                     affinity='precomputed',
                                                     linkage='single'))
            cluster_s[i].fit_predict(dm[i])
            cluster_s[i].labels_ = cluster_s[i].labels_ + 1
            sfS.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_s[i].labels_))

            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_s[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue

        try:
            cluster_spec.append(SpectralClustering(n_clusters=clus[i], affinity="rbf", assign_labels="discretize"))
            cluster_spec[i].fit_predict(dm[i])
            cluster_spec[i].labels_ = cluster_spec[i].labels_ + 1
            pred_actual[i] = np.array(pred_actual[i])
            sfSpec.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_spec[i].labels_))
            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_spec[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue
        try:
            cluster_alg.append(AgglomerativeClustering(n_clusters=clus[i]))
            cluster_alg[i].fit_predict(dm[i])
            cluster_alg[i].labels_ = cluster_alg[i].labels_ + 1
            pred_actual[i] = np.array(pred_actual[i])

            sfAlg.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_alg[i].labels_))

            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_alg[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue

        try:
            cluster_ap.append(AffinityPropagation(random_state=0, verbose=True))

            cluster_ap[i].fit_predict(dm[i])

            cluster_ap[i].labels_ = cluster_ap[i].labels_ + 1
            pred_actual[i] = np.array(pred_actual[i])
            sfAp.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_ap[i].labels_))
            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_ap[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue
        try:
            cluster_p.append(KMedoids(n_clusters=clus[i], random_state=0))
            cluster_p[i].fit_predict(dm[i])
            cluster_p[i].labels_ = cluster_p[i].labels_ + 1
            pred_actual[i] = np.array(pred_actual[i])
            sfKmed.append((sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_p[i].labels_)))
            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], cluster_p[i].labels_), 4)))
            f.write("\n")
        except:
            f.write("0")
            f.write("\n")
            continue

        ######## End Unsupervised Learning  #######

        semi_d, lb = semiData_Gen("datasets/", val, "y", dataRatio=float(dir))
        SmD = SemiDf(semi_d, lb)
        semi_df.append(semi_d)
        lbl.append(lb)
        SmDt.append(SmD)

        ## see the explanation above where y argument declared
        # y[0].leafs
        # y[1].indleaf

        y.append(leaf_Weight_helper(RF[i], SmDt[i]))

        # assign wieght to leaf
        tl.append(Tree_leaf_weight(y[i], SmDt[i], clus[i]))
        # declare proximity matrix
        pmt.append(np.zeros((len(lbl[i]), len(lbl[i]))))
        # calculate proximity matrix with weightage

        pmt2.append(get_weighted_prm(tl[i], pmt[i]))
        pmt2[i] = 1 - pmt2[i]
        pmt2[i] = preprocessing.normalize(pmt2[i])

        """
        Image Clustring if we will see proximity
        matrix given the number of objects cluster the proximity matrix

        another idea : Use GANS to generate dataset of 

        proxmity matrix and train classifier on it

        helpful recources:
        Document
        https://www.diva-portal.org/smash/get/diva2:1119048/FULLTEXT01.pdf
        Coding
        https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python


        """

        ## Clustering

        # f.write("\n Semi Supervised \n")

        try:
            s_cluster_c.append(AgglomerativeClustering(n_clusters=clus[i],
                                                       affinity='precomputed',
                                                       linkage='complete'))
            s_cluster_c[i].fit_predict(pmt2[i])
            s_sfC.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_c[i].labels_))
            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_c[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue

        try:
            s_cluster_s.append(AgglomerativeClustering(n_clusters=clus[i],
                                                       affinity='precomputed',
                                                       linkage='single'))
            s_cluster_s[i].fit_predict(pmt2[i])
            s_sfS.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_s[i].labels_))

            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_s[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue

        try:
            s_cluster_spec.append(SpectralClustering(n_clusters=clus[i], affinity="rbf", assign_labels="discretize"))
            s_cluster_spec[i].fit_predict(pmt2[i])
            pred_actual[i] = np.array(pred_actual[i])
            s_sfSpec.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_spec[i].labels_))

            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_spec[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue

        try:
            s_cluster_alg.append(AgglomerativeClustering(n_clusters=clus[i]))
            s_cluster_alg[i].fit_predict(pmt2[i])
            pred_actual[i] = np.array(pred_actual[i])

            s_sfAlg.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_alg[i].labels_))

            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_alg[i].labels_), 4)))
            f.write(",")
        except:
            f.write("0.0")
            f.write(",")
            continue

        try:
            s_cluster_ap.append(AffinityPropagation(random_state=0, verbose=True))
            s_cluster_ap[i].fit_predict(pmt2[i])
            pred_actual[i] = np.array(pred_actual[i])
            s_sfAp.append(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_ap[i].labels_))
            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_ap[i].labels_), 4)))
            f.write(",")
        except:
            print("Exception")
            f.write("0.0")
            f.write(",")
            continue
        try:
            s_cluster_p.append(KMedoids(n_clusters=clus[i], random_state=0))
            s_cluster_p[i].fit_predict(pmt2[i])
            pred_actual[i] = np.array(pred_actual[i])
            s_sfKmed.append((sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_p[i].labels_)))
            f.write(str(round(sklearn.metrics.adjusted_rand_score(pred_actual[i], s_cluster_p[i].labels_), 4)))
            f.write("\n")
        except:
            f.write("0.0")
            f.write("\n")
            continue

        #################End 0.5 ratio###

        i = i + 1
        f.write("\n")

    # unsupervised proximities
    # display_proximities(dm, 3, 2)

    # semi supervised proximities

    # display_proximities(pmt, 3, 2)

    f.close()


# Press the green button in the gutter to run the script.
from pandas.plotting import table
from openpyxl import Workbook


def createplot(desc, i, k):
    # create a subplot without frame
    plot = plt.subplot(111, frame_on=False)

    # remove axis
    plot.xaxis.set_visible(False)
    plot.yaxis.set_visible(False)

    # create the table plot and position it in the upper left corner
    table(plot, desc, loc='upper right')

    # save the plot as a png file
    plt.savefig(f'desc_plot{k}{i}.png')


def avg_cal(path, si, ei):
    df2 = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=list('ABCDEF'))
    for j in range(0, 20):
        f = open(f'{path}/storage{j}', "r")

        line = f.readlines()

        for j in range(si, ei):
            line[j] = line[j].replace('\n', '')
            line[j] = line[j].split(",")
            x = np.array(line[j])
            y = x.astype(np.float)
            df = pd.DataFrame([y], columns=list('ABCDEF'))
            df2 = df2.append(df, ignore_index=True)
    return df2


if __name__ == '__main__':
    #arr = "Lung.csv"

    #####------new-------
    dr = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    # dr = ["0.3"]

    for k in dr:
        for j in range(0, 20):
            unsuperv_rf_proximity('PyCharm', j, k)

        # This  will create file in following format
        # Datsets
        # unsupervised Cluster
        # 0.1 Semisupervised

    #####------new-------
    inds = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, 34]

    # inds = [1, 2]
    df1 = pd.DataFrame(columns=["K"])

    for k in dr:
        # print("\n ===Ratio :", k)
        # print("====\n")
        for j in inds:
            df = avg_cal(k, j, j + 1)
            df = df.iloc[1:]
            # print("=====")

            # print(df)
            # print(df)
            # display_all(df.describe().T)
            # print("======")
            # print(df.describe().T["mean"])
            if j == 1:
                df1 = pd.concat(
                    [pd.DataFrame([round(df.describe().T["mean"][i], 4)], columns=[str(j)]) for i in range(6)],
                    ignore_index=True)
            else:
                df2 = pd.DataFrame(columns=[str(j)])
                df2 = pd.concat(
                    [pd.DataFrame([round(df.describe().T["mean"][i], 4)], columns=[str(j)]) for i in range(6)],
                    ignore_index=True)
                df1 = df1.join(df2)
        df1.to_excel(f'output {k}.xlsx')


























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

# createplot(df.describe(),j,k)


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

# calculate_avg()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
