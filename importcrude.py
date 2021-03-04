import pandas
import numpy

def string_to_list(string):
    
    s = string[0].split(",")
    
    myNums = [float(elem) for elem in s]

    return myNums


def import_crudeOil():
    
    datar = pandas.read_csv("crude-oil.csv", header=None)
    V = datar.values
    
    
    all_data = []
    
    for i in range(len(V)):
        all_data.append(string_to_list(V[i]))
  
    data = []
    targetf = []
        
    for i in range(len(all_data)):
        data.append(all_data[i][0:-1])
        targetf.append(all_data[i][5])
    
    target = []
    
    for num in targetf:
        target.append(int(num))
    
    class datafull:
        data = 0
        target = 0
        
    datafull.data = numpy.asarray(data)
    datafull.target = numpy.asarray(target)
    
    return datafull

