import pandas as pd
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_analysis_table_cluster_percentages_avg(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')


    df1 = data.iloc[:, 2]

    for i in range(4,20,2):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.mean(axis=1)


def make_analysis_table_cluster_percentages_max(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')


    df1 = data.iloc[:, 2]

    for i in range(4,20,2):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.max(axis=1)


def make_analysis_table_cluster_percentages_avg_unsup(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')
    print(data)

    df1 = data.iloc[:, 1]
    print(df1)

    for i in range(3,20,3):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.mean(axis=1)


def make_analysis_table_cluster_percentages_sup(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')


    df1 = data.iloc[:, 2]

    for i in range(4,20,2):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.mean(axis=1)



def make_analysis_table_cluster_percentages_max(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')


    df1 = data.iloc[:, 2]

    for i in range(4,20,2):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.max(axis=1)

def make_analysis_table_cluseter_unsup(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')


    df1 = data.iloc[:, 2]

    for i in range(4,20,2):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.mean(axis=1)

def make_analysis_table_cluster_percentages_avg_unsup(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')


    df1 = data.iloc[:, 1]
    #print(df1)
    for i in range(3,20,3):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.mean(axis=1)

def make_analysis_table_cluster_percentages_avg_sup(filename):
    """
Average unsuper only
"""


    data = pd.read_excel(filename, sheet_name='Sheet1')

    df1 = data.iloc[:, 1]
    for i in range(2,10):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.mean(axis=1)

def make_analysis_table_cluster_percentages_max_sup(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')

    df1 = data.iloc[:, 1]
    for i in range(2,10):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.max(axis=1)

def sup_dataset_max(filename):
    data = pd.read_excel(filename, sheet_name='Sheet1')

    df1 = data.iloc[0, 1:]

    for i in range(1, 6):
        df2 = data.iloc[i, 1:]
        df1 = pd.concat([df1, df2], axis=1)

    return df1.max(axis=1)

def unsup_dataset_max(filename):
    data = pd.read_excel(filename, sheet_name='Sheet1')

    df1 = data.iloc[0, 1:]

    for i in range(1, 6):
        df2 = data.iloc[i, 1:]
        df1 = pd.concat([df1, df2], axis=1)
    #print("\n-------max--------\n",df1)
    return df1.max(axis=1)

def sup_dataset_avg(filename):
    data = pd.read_excel(filename, sheet_name='Sheet1')

    df1 = data.iloc[0, 1:]

    for i in range(1, 6):
        df2 = data.iloc[i, 1:]
        df1 = pd.concat([df1, df2], axis=1)

    return df1.mean(axis=1)
def unsup_dataset_avg(filename):


    data = pd.read_excel(filename, sheet_name='Sheet1')

    df1 = data.iloc[0, 1:]

    for i in range(1, 6):
        df2 = data.iloc[i, 1:]
        df1 = pd.concat([df1, df2], axis=1)

    return df1.mean(axis=1)


def unsup_dataset_graph(col,ind,filename):
    d1 = unsup_dataset_avg(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_avg(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_avg(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_avg(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_avg(filename+'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    #d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d=d.reset_index(drop=True)
    #print(ind)

    d = d[np.arange(len(d)) % 2 == 1]
    print(d.mean(axis=1))

    f = d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3], 9:ind[4] ,11:ind[5], 13:ind[6],15:ind[7],17:ind[8]})

    #d.set_names(ind, inplace=True)


    #print(ind,d.index.names)
    #print(d)
    print("#################")
    d1 = unsup_dataset_max(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_max(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_max(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_max(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_max(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    #d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d=d.reset_index(drop=True)
    #d_o = d[np.arange(len(d)) % 2 == 0]
    d = d[np.arange(len(d)) % 2== 1]
    #print(d_o.max(axis=1))
    d= d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3],9:ind[4] ,11:ind[5],13:ind[6],15:ind[7],17:ind[8]})
    #d.index.rename("Dataset", inplace=True)
    print("hey",d)
    col2=col
    #print(col,col[1:])
    f1,df1=unsup_dataset_graph2(["0.1","0.3","0.5","0.7","0.9"],ind,filename)
    print("hey 2",df1)
    f=pd.concat([f1,f],axis=1)
    d=pd.concat([df1,d],axis=1)
    f.columns=["0.0","0.1","0.3","0.5","0.7","0.9"]
    d.columns=["0.0","0.1","0.3","0.5","0.7","0.9"]

    print("f------",f,"\ndf-------",d)
    return f,d
    #print(d)
    print("#################")






def unsup_dataset(col,ind,filename):
    d1 = unsup_dataset_avg(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_avg(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_avg(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_avg(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_avg(filename+'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
        d.iloc[:, c] =  d.iloc[:,c].astype(str) + '&'
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].astype(str) +"\\"
    d=d.reset_index(drop=True)
    #print(ind)

    #d_o = d[np.arange(len(d)) % 2 == 0]
    d = d[np.arange(len(d)) % 2== 1]
    #print(d_o.mean(axis=1))
    print("  \hline Dataset &",d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3], 9:ind[4] ,11:ind[5], 13:ind[6],15:ind[7],17:ind[8]}),"\\" )
    #d.set_names(ind, inplace=True)


    #print(ind,d.index.names)
    #print(d)
    print("#################")
    d1 = unsup_dataset_max(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_max(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_max(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_max(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_max(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
        d.iloc[:, c] = d.iloc[:, c].astype(str) + '&'
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].astype(str) + "\\"
    d=d.reset_index(drop=True)
    #d_o = d[np.arange(len(d)) % 2 == 0]
    d = d[np.arange(len(d)) % 2== 1]
    #print(d_o.max(axis=1))
    print("  \hline Dataset &",d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3],9:ind[4] ,11:ind[5],13:ind[6],15:ind[7],17:ind[8]}),"\\" )
    #d.index.rename("Dataset", inplace=True)

    #print(d)
    print("#################")



def unsup_dataset_graph2(col,ind,filename):
    d1 = unsup_dataset_avg(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_avg(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_avg(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_avg(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_avg(filename+'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d=d.reset_index(drop=True)
    #print(ind)

    #d_o = d[np.arange(len(d)) % 2 == 0]
    d = d[np.arange(len(d)) % 2== 0]
    f= d.mean(axis=1)
    f=f.rename(index={0: ind[0], 2: ind[1],4: ind[2],6: ind[3],8:ind[4] ,10:ind[5],12:ind[6],14:ind[7],16:ind[8]})

    #f = pd.concat([d.mean(axis=1),d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3], 9:ind[4] ,11:ind[5], 13:ind[6],15:ind[7],17:ind[8]})],axis=1)
    #print(f)

    #print(ind,d.index.names)
    #print(d)
    print("#################")
    d1 = unsup_dataset_max(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_max(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_max(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_max(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_max(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d=d.reset_index(drop=True)
    #d_o = d[np.arange(len(d)) % 2 == 0]
    d = d[np.arange(len(d)) % 2== 0]
    df=d.max(axis=1)
    df=df.rename(index={0: ind[0], 2: ind[1],4: ind[2],6: ind[3],8:ind[4] ,10:ind[5],12:ind[6],14:ind[7],16:ind[8]})
    #df=pd.concat([,d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3],9:ind[4] ,11:ind[5],13:ind[6],15:ind[7],17:ind[8]})],axis=1)
    #d.index.rename("Dataset", inplace=True)

    #print(d)

    print("f", f,"\n", "df",df)
    return f,df
    print("#################")







def unsup_dataset_2(col,ind,filename):
    d1 = unsup_dataset_avg(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_avg(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_avg(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_avg(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_avg(filename+'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d=d.reset_index(drop=True)
    #print(ind)

    #d_o = d[np.arange(len(d)) % 2 == 0]
    d = d[np.arange(len(d)) % 2== 0]
    print(d.mean(axis=1))
    print("  \hline Dataset &",d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3], 9:ind[4] ,11:ind[5], 13:ind[6],15:ind[7],17:ind[8]}),"\\" )
    #d.set_names(ind, inplace=True)


    #print(ind,d.index.names)
    #print(d)
    print("#################")
    d1 = unsup_dataset_max(filename+'output 0.1.xlsx')
    d2 = unsup_dataset_max(filename+'output 0.3.xlsx')
    d3 = unsup_dataset_max(filename+'output 0.5.xlsx')
    d4 = unsup_dataset_max(filename+'output 0.7.xlsx')
    d5 = unsup_dataset_max(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d=d.reset_index(drop=True)
    #d_o = d[np.arange(len(d)) % 2 == 0]
    d = d[np.arange(len(d)) % 2== 0]
    print(d.max(axis=1))
    print("  \hline Dataset &",d.rename(index={1: ind[0], 3: ind[1],5: ind[2],7: ind[3],9:ind[4] ,11:ind[5],13:ind[6],15:ind[7],17:ind[8]}),"\\" )
    #d.index.rename("Dataset", inplace=True)

    #print(d)
    print("#################")




def sup_dataset(col,ind,filename):
    d1 = sup_dataset_avg(filename+'output 0.1.xlsx')
    d2 = sup_dataset_avg(filename+'output 0.3.xlsx')
    d3 = sup_dataset_avg(filename+'output 0.5.xlsx')
    d4 = sup_dataset_avg(filename+'output 0.7.xlsx')
    d5 = sup_dataset_avg(filename+'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
        d.iloc[:, c] =  d.iloc[:,c].astype(str) + '&'
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].astype(str) +"\\"
    d=d.reset_index(drop=True)
    #print(ind)
    print(" \_begin {table}[H] \_title{New Avg} \centering \_begin{tabular}{|l|c|c|c|c|c|r|} \hline Dataset &",d.rename(index={0: ind[0], 1: ind[1],2: ind[2],3: ind[3],4: ind[4],5:ind[5] ,6:ind[6],7:ind[7],8:ind[8]}),"\\ \end{tabular} \caption{states all complete and single clustering methods have not shown improvement in evaluation measure.However Hierarchical clustering have shown significant improvement.} \label{tab:my_label} \end{table}")
    #d.index.rename("Dataset", inplace=True)
    #d.set_names(ind, inplace=True)


    #print(ind,d.index.names)
    #print(d)
    print("#################")
    d1 = sup_dataset_max(filename+'output 0.1.xlsx')
    d2 = sup_dataset_max(filename+'output 0.3.xlsx')
    d3 = sup_dataset_max(filename+'output 0.5.xlsx')
    d4 = sup_dataset_max(filename+'output 0.7.xlsx')
    d5 = sup_dataset_max(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
        d.iloc[:, c] = d.iloc[:, c].astype(str) + '&'
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].astype(str) + "\\"
    d=d.reset_index(drop=True)

    print("\_begin{table}[H] \_title{New Mzx} \centering \_begin{tabular}{|l|c|c|c|c|c|r|} \hline Dataset &",d.rename(index={0: ind[0], 1: ind[1],2: ind[2],3: ind[3],4: ind[4],5:ind[5] ,6:ind[6],7:ind[7],8:ind[8]}),"\\ \end{tabular} \caption{states all complete and single clustering methods have not shown improvement in evaluation measure.However Hierarchical clustering have shown significant improvement.} \label{tab:my_label} \end{table}")
    #d.index.rename("Dataset", inplace=True)

    #print(d)
    print("#################")

def sup_dataset_graph(col,ind,filename):
    d1 = sup_dataset_avg(filename+'output 0.1.xlsx')
    d2 = sup_dataset_avg(filename+'output 0.3.xlsx')
    d3 = sup_dataset_avg(filename+'output 0.5.xlsx')
    d4 = sup_dataset_avg(filename+'output 0.7.xlsx')
    d5 = sup_dataset_avg(filename+'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d=d.reset_index(drop=True)
    #print(ind)
    f = d.rename(index={0: ind[0], 1: ind[1],2: ind[2],3: ind[3],4: ind[4],5:ind[5] ,6:ind[6],7:ind[7],8:ind[8]})
    #d.set_names(ind, inplace=True)


    #print(ind,d.index.names)
    #print(d)
    print("#################")
    d1 = sup_dataset_max(filename+'output 0.1.xlsx')
    d2 = sup_dataset_max(filename+'output 0.3.xlsx')
    d3 = sup_dataset_max(filename+'output 0.5.xlsx')
    d4 = sup_dataset_max(filename+'output 0.7.xlsx')
    d5 = sup_dataset_max(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d=d.reset_index(drop=True)

    d = d.rename(index={0: ind[0], 1: ind[1],2: ind[2],3: ind[3],4: ind[4],5:ind[5] ,6:ind[6],7:ind[7],8:ind[8]})
    #d.index.rename("Dataset", inplace=True)

    #print(d)

    return f,d
    print("#################")





def sup_clus_graph(col,ind,filename):
    d1 = make_analysis_table_cluster_percentages_avg_sup(filename+'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_avg_sup(filename+'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_avg_sup(filename+'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_avg_sup(filename+'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_avg_sup(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d= d.rename(index={0: ind[0], 1: ind[1],2: ind[2],3: ind[3],4: ind[4],5:ind[5]} )
    d.index.rename("Cluster", inplace=True)
    #d.set_names(ind, inplace=True)
    f=d

    print(ind,d.index.names)
    print(d)
    print("#################")

    d1 = make_analysis_table_cluster_percentages_max_sup(filename+'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_max_sup(filename+'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_max_sup(filename+'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_max_sup(filename+'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_max_sup(filename+'output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d=d.rename(index={0: ind[0], 1: ind[1], 2: ind[2], 3: ind[3], 4: ind[4], 5: ind[5]})
    d.index.rename("Cluster", inplace=True)

    print(d)
    return f,d
    print("#################")


def sup_clus(col,ind):
    d1 = make_analysis_table_cluster_percentages_avg_sup('output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_avg_sup('output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_avg_sup('output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_avg_sup('output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_avg_sup('output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1]-1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].round(3)
    d.iloc[:, d.shape[1]-1] = d.iloc[:, d.shape[1]-1].astype(str) +"\\"
    print("\hline Cluster &",d.rename(index={0: ind[0], 1: ind[1],2: ind[2],3: ind[3],4: ind[4],5:ind[5]} ),"\\")
    d.index.rename("Cluster", inplace=True)
    #d.set_names(ind, inplace=True)


    print(ind,d.index.names)
    print(d)
    print("#################")

    d1 = make_analysis_table_cluster_percentages_max_sup('output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_max_sup('output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_max_sup('output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_max_sup('output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_max_sup('output 0.9.xlsx')
    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
        d.iloc[:, c] = d.iloc[:, c].astype(str) + '&'
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].astype(str) + "\\"
    print("\hline Cluster &", d.rename(index={0: ind[0], 1: ind[1], 2: ind[2], 3: ind[3], 4: ind[4], 5: ind[5]}), "\\")
    d.index.rename("Cluster", inplace=True)

    print(d)
    print("#################")


def unsup(col,ind,path):
    # UnSupervised Only average
    d1 = make_analysis_table_cluster_percentages_avg_unsup(path + 'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_avg_unsup(path +'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_avg_unsup(path + 'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_avg_unsup(path +'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_avg_unsup(path +'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    lm = d.max(axis=1)
    l=d.mean(axis=1)
    print(d.mean(axis=1))
    print("#################")
    d1 = make_analysis_table_cluster_percentages_avg(path +'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_avg(path +'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_avg(path +'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_avg(path +'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_avg(path +'output 0.9.xlsx')


    d = pd.concat([l,d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
        d.iloc[:, c] = d.iloc[:, c].astype(str) + '&'
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].astype(str) + "\\"
    print("\hline Cluster &", d.rename(index={0: ind[0], 1: ind[1], 2: ind[2], 3: ind[3], 4: ind[4], 5: ind[5]}), "\\")
    d.index.rename("Cluster", inplace=True)

    print(d)
    # print(d.mean(axis=1))

    print("Max")
    print("#################")
    d1 = make_analysis_table_cluster_percentages_max(path +'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_max(path +'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_max(path +'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_max(path +'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_max(path +'output 0.9.xlsx')

    d = pd.concat([lm,d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
        d.iloc[:, c] = d.iloc[:, c].astype(str) + '&'
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].astype(str) + "\\"
    print("\hline Cluster &", d.rename(index={0: ind[0], 1: ind[1], 2: ind[2], 3: ind[3], 4: ind[4], 5: ind[5]}), "\\")
    d.index.rename("Cluster", inplace=True)

    print(d)
    return d



def unsup_graph(col,ind,path):
    # UnSupervised Only average
    d1 = make_analysis_table_cluster_percentages_avg_unsup(path + 'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_avg_unsup(path +'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_avg_unsup(path + 'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_avg_unsup(path +'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_avg_unsup(path +'output 0.9.xlsx')

    d = pd.concat([d1, d2, d3, d4, d5], axis=1)
    lm = d.max(axis=1)
    l=d.mean(axis=1)
    print(d.mean(axis=1))
    print("#################")
    d1 = make_analysis_table_cluster_percentages_avg(path +'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_avg(path +'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_avg(path +'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_avg(path +'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_avg(path +'output 0.9.xlsx')


    d = pd.concat([l,d1, d2, d3, d4, d5], axis=1)

    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d = d.rename(index={0: ind[0], 1: ind[1], 2: ind[2], 3: ind[3], 4: ind[4], 5: ind[5]})
    d.index.rename("Cluster", inplace=True)
    f=d
    print("ffksdksjdksjdks#######",d,"dfndkfndskl",d.columns)
    # print(d.mean(axis=1))

    print("Max")
    print("#################")
    d1 = make_analysis_table_cluster_percentages_max(path +'output 0.1.xlsx')
    d2 = make_analysis_table_cluster_percentages_max(path +'output 0.3.xlsx')
    d3 = make_analysis_table_cluster_percentages_max(path +'output 0.5.xlsx')
    d4 = make_analysis_table_cluster_percentages_max(path +'output 0.7.xlsx')
    d5 = make_analysis_table_cluster_percentages_max(path +'output 0.9.xlsx')

    d = pd.concat([lm,d1, d2, d3, d4, d5], axis=1)
    d.columns = col
    for c in range(d.shape[1] - 1):
        d.iloc[:, c] = d.iloc[:, c].round(3)
    d.iloc[:, d.shape[1] - 1] = d.iloc[:, d.shape[1] - 1].round(3)
    d=d.rename(index={0: ind[0], 1: ind[1], 2: ind[2], 3: ind[3], 4: ind[4], 5: ind[5]})
    d.index.rename("Cluster", inplace=True)

    print("d#######",d)
    return f,d


if __name__ == '__main__':
    ni= "Proposal1-Clus-avg"
    """
    f,df= unsup_dataset_graph(["0.0","0.1", "0.3", "0.5", "0.7", "0.9"],
                  ["Breast-tissues", "Auto Mpg", "Lung", "Parkinson",
                   "Iris", "Glass", "Heart", "Wbc", "Wine"],"proposal_1/unsup/")
    f2,df2=sup_dataset_graph(["0.1", "0.3", "0.5", "0.7", "0.9"],["Breast-tissues",
                                                                  "Auto Mpg" ,
                                                                  "Lung" ,
                                                                  "Parkinson",
                                                                  "Iris",
                                                                  "Glass",
                                                                  "Heart",
                                                                  "Wbc",
                                                                  "Wine"],"proposal_1/semi/")



    """

    #unsup(["0.1& ","0.3&","0.5&","0.7&","0.9\\"])
    f, df = unsup_graph(["0.0","0.1","0.3","0.5","0.7","0.9"], ["Complete"," Single ","Spectral","Hierarical","Affinity Propagation","K-Mediods"],
                        "proposal_2/unsup/")
    #print("#1\n",df)
    f2, df2 = sup_clus_graph(["0.1","0.3","0.5","0.7","0.9"], ["Complete"," Single ","Spectral","Hierarical","Affinity Propagation","K-Mediods"],
                             "proposal_2/semi/")
    print(f,df)
    f2 = f2.dropna()
    df2 =df2.dropna()
    f=f.dropna()
    df = df.dropna()

    print("avg",f, "max",df)
    #sup_clus_graph(["0.1 ","0.3","0.5","0.7","0.9"], ["Complete ","\hline Single &","\hline Spectral &","\hline Hierarical &","\hline Affinity Propagation &","\hline K-Mediods &"])
    #sup_dataset(["0.1& ", "0.3&", "0.5&", "0.7&", "0.9\\"],["\hline Breast-tissues &",   "\hline Auto Mpg &" ,"\hline Lung &" ,"\hline Parkinson &  ","\hline Iris &","\hline Glass &","\hline Heart &","\hline Wbc &","\hline Wine &"])
    """
    unsup_dataset_graph(["0.1",
                   "0.3",
                   "0.5",
                   "0.7",
                   "0.9"],["Breast-tissues",
                           "Auto Mpg" ,
                           "Lung" ,
                           "Parkinson",
                           "Iris",
                           "Glass",
                           "Heart",
                           "Wbc",
                           "Wine"],
                  "proposal_2/unsup/")   
    print("Hiii..----")
    unsup_dataset_2(["0.1& ", "0.3&", "0.5&", "0.7&", "0.9\\"],["\hline Breast-tissues &",   "\hline Auto Mpg &" ,"\hline Lung &" ,"\hline Parkinson &  ","\hline Iris &","\hline Glass &","\hline Heart &","\hline Wbc &","\hline Wine &"],"proposal_2/unsup/")
    
    """

    #df = make_analysis_table_cluster_percentages_avg('proposal_1/unsup/output 0.1.xlsx')
    #print(df,"#2")
    #print(df.columns)
    #threedee = plt.figure().gca(projection='3d')
    #threedee.scatter(df.index, df[df.columns[1]], df[df.columns[2]])
    #threedee.set_xlabel('Index')
    #threedee.set_ylabel('H-L')
    #threedee.set_zlabel('Close')
    #plt.show()

    print(df,"\n",df2)
    markers = [".", ",", "o", "v", "^", "<", ">"]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    thisLegend = []
    markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']

    #plt.use('Agg')  # avoid displaying the figure in Python environment
    i=0

    #plt.plot(df2["0.1"], df2.index, markers[4], c=colors[1],
             #label="marker='{0}'".format(markers[4]))
    #plt.show()
    for thisAlpha in df.columns:
        if i == 0:
            #plt.scatter(df[thisAlpha], df.index,c=colors[0], s=30,marker=markers[0], alpha=float(thisAlpha));  # -. represents dashed-dotted line-style
            plt.plot(df[thisAlpha], df.index, markers[0],c=colors[0],
                     label="marker='{0}'".format(markers[0]))
            thisLegend.append("Method1 = {}".format(thisAlpha))
        else:
            plt.plot(df[thisAlpha], df.index, markers[1],c=colors[0],
                     label="marker='{0}'".format(markers[1])) # -. represents dashed-dotted line-style
            thisLegend.append("Method1 = {}".format(thisAlpha))
            plt.plot(df2[thisAlpha], df2.index, markers[4],c=colors[1],
                     label="marker='{0}'".format(markers[4]))
            thisLegend.append("Method2 = {}".format(thisAlpha))
        i=i+1

    plt.legend(thisLegend,loc= "best");
    plt.savefig(f"{ni}.png")
    plt.show()


#print("Hello ",sup_dataset_avg("output 0.1.xlsx"))