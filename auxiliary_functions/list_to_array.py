import numpy

def list_to_array(list):
    list_array = numpy.zeros((len(list)-1, len(list[0])))
    for i in range(list_array.shape[0]):
        for j in range(list_array.shape[1]):
            list_array[i][j] = list[i+1][j]
    list_array = list_array.astype(int)
    return list_array
