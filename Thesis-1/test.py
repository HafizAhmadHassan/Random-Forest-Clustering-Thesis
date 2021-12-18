import math
import numpy as np

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
    print(arr)
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
    print(ml)
    print(arr)


    """
    check if nan is present
    
    
    
    array_sum = np.sum(array)
    array_has_nan = np.isnan(array_sum)
    
    
    """
    ml2=[]
    array_sum = np.sum(ml)
    array_has_nan = np.isnan(array_sum)
    if(array_has_nan):
        #print(array_has_nan)
        for u in range(0,len(ml)):
            if(np.isnan(ml[u])):
                continue
            else:
                ml2.append(ml[u])
    #print("this is ml2",ml2)
    ml = np.array(ml, dtype=float)
    s = np.sum(arr)

    a = arr[k+1] / s

    b = ((s - arr[k+1])/s) * (1 + np.var(ml2))

    # b = (s - arr[k+1])/ s) * (1 + np.var(ml2)) good result
    #(s - arr[k+1])/s * (1 + np.var(ml2)) close to entropy
    if(np.isnan(b) and a == 1.0):
        return a
    return a * b

def pre_process_scl(ml, k):
    arr = []
    arr2= []
    for i in range(0, k + 2):
        arr.append(0)

    for i in range(0, len(ml)):
        if (math.isnan(ml[i])):
            arr[k + 1] = arr[k + 1] + 1
        else:
            arr[ml[i]] = arr[ml[i]] + 1

    for i in range(0, k + 2):
        arr2.append(0)


    return arr2


import numpy as np


def rescaling(arr):
    arr = np.array(arr)
    arr1 = np.sort(arr, axis=None)
    print(arr1)
    for i in range(0, len(arr1) - 1):
        print("yes")
        s = arr1[i+1] - arr1[i]
        print(s)
        if (s > 1):
            arr1[i + 1] = arr1[i + 1] - (s - 1)
    return arr1



if __name__ == '__main__':
    print(rescaling( [np.nan,2,3,5,3,6]))

    #print(compute_entropy([0, 0, 0, 0, 0, 0, 0, 1,np.NAN],5))
    #print(compute_weigth([0, 0, 0, 0, 0, 0, 0, 1,np.NAN],5))

    #print("--------")
    #print(compute_entropy([np.NAN], 2))
    #print(compute_weigth([np.NAN], 2))

    #print("--------")
    #print(compute_entropy([], 5))
    #print(compute_weigth([], 5))

    print("--------")
    print(np.var([0, 0, 0, 0, 0, 0, 0, 1]))
    print(np.var([1, 1, 1, 1, 1, 1, 1, 3]))
