import pandas as pd
import numpy as np
def make_analysis_table(filename):

    data = pd.read_excel(filename, sheet_name='Sheet1')


    df1 = data.iloc[:, 2]

    for i in range(4,20,2):
        df2 = data.iloc[:, i]
        df1 = pd.concat([df1,df2],axis=1)

    return df1.mean(axis=1)

if __name__ == '__main__':
    d1 = make_analysis_table('Output 0.1.xlsx')
    d2 = make_analysis_table('Output 0.3.xlsx')
    d3 =make_analysis_table('Output 0.5.xlsx')
    d4 =make_analysis_table('Output 0.7.xlsx')
    d5 =make_analysis_table('Output 0.9.xlsx')

    d = pd.concat([d1,d2,d3,d4,d5],axis=1)
    print(d)

