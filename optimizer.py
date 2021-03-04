import WOA as woa #Importa o algoritmo WOA
import CS as cs #Importa o algoritmo CS
import CSO as cso #Importa o algoritmo CSO
import benchmarks #Importa as funções para otimização
import numpy
import tempfile
import pandas


#Importa as funções métricas
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances
import functions
import methods

#Importa os Datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets.mldata import fetch_mldata
import tempfile
import importcrude

from scipy.stats import mode
from munkres import Munkres
import math
import copy
import csv
import numpy
import time
import matplotlib.pyplot as plt

def compute_err(bd,od,nk):

    xor = numpy.zeros((nk,nk))
    for i in range (nk):
        for j in range(nk):
            tmp1 = float(numpy.logical_and((bd==i) , (od==j)).sum())
            tmp2 = float((od==j).sum())
            xor[i,j]= tmp2-tmp1
    return xor

def labelmatch(gold,predict,nk):

    #denom = float((gold!=0).sum())
    size = gold.shape
    cost  = compute_err(gold,predict,nk)
    out = numpy.zeros((size))
    m = Munkres()
    indexes = m.compute(cost)
    for row,col in indexes:
        inds  = (predict == col).nonzero()
        out[inds] = row
    return (out)

def rotulos(particle, K , dataset):
    etiqueta = 0
    rotulo = numpy.zeros(data.namostras)
    for i in range(0,data.namostras):
        melhor = float("inf")
        for r in range(0,K):
            distancia = 0
            for k in range(0,data.ndim):
                distancia = distancia + pow (particle[r][k] - dataset[i][k],2)
            distancia = math.sqrt(distancia)
            if (distancia < melhor):
                melhor = distancia
                etiqueta = r

        rotulo[i] = etiqueta
    return rotulo

#datafull = fetch_mldata('glass', data_home=test_data_home)
#datafull = fetch_mldata('uci-20070111 wine', data_home=test_data_home)
datafull = load_iris()
#datafull = load_breast_cancer()
#datafull = importcrude.import_crudeOil()

dataset = datafull.data

class Data:
    namostras = 0
    ndim = 0
    ncluster = 0

data = Data()
    
data.ncluster = len(set(datafull.target))
    
data.namostras = dataset.shape[0]

data.ndim = dataset.shape[1]

dim = data.ndim

lb = []
ub = []

y_true = copy.copy(datafull.target)

for i in range(dim):
    lb.append(dataset[:,i].min())
    ub.append(dataset[:,i].max())

def selector(algo,func_details,popSize,Iter,lb,ub,dim,cluster):
    function_name=func_details[0]
    if(algo==0):
        x=woa.WOA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,data,dataset,cluster)
    if(algo==1):
        x=cs.CS(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,data,dataset,cluster)
    if(algo==2):
        x=cso.CSO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,data,dataset,cluster)
    return x
    
    
# Seleção dos métodos de otimização

WOA= False
CS = True
CSO = False


# Seleção das funções de otimização

distances=True

optimizer=[WOA, CS, CSO]
benchmarkfunc=[distances] 
        
# Selecão do número de repetições para cada experimento
#EUCLIDIANA

NumOfRuns = 10

NumOfClusters=10

# Seleção de parâmetros gerais para todos os otimizadores
PopulationSize = 50

Iterations= 500
Export=True

#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
#ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))
    
limits = numpy.empty([NumOfClusters,2])
otimo = numpy.empty([NumOfClusters])

cft = []
f1t = []
accurracyt = []
silhouettet = []
davisbouldint = []
dunnt = []
arit = []
hbt = []
amit = []

Kmedio = 0
fitness1 = 0
cft1 = 0
f1t1 = 0
accurracyt1 = 0
silhouettet1 = 0
davisbouldint1 = 0
dunnt1 = 0
arit1 = 0
hbt1 = 0
amit1 = 0

conthbt = NumOfRuns
contsilh= NumOfRuns
contdb = NumOfRuns

for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i]==True) and (benchmarkfunc[j]==True)):
            for n in range(0,NumOfRuns):                          
                for k in range (2,NumOfClusters+2):
                
                    print()
                    print(k-1)
                    
                    func_details=benchmarks.getFunctionDetails(j)
                    x=selector(i,func_details,PopulationSize,Iterations,lb,ub,dim,k)
                    
                    otimo[k-2] = x.last
                    
                    predict = rotulos(x.bestIndividual,k,dataset)
                    labels = labelmatch(datafull.target,predict,data.ncluster)
                    y_pred = labels
                    
                    cft.append(confusion_matrix(y_true, y_pred))
                    
                    if(len(set(y_pred)) != 1):
                        hbt.append(calinski_harabaz_score(dataset,y_pred))
                    else:
                        hbt.append(0)
                       
                        
                    arit.append(adjusted_rand_score(y_true, y_pred))
                    
                    amit.append(adjusted_mutual_info_score(y_true, y_pred))
                    
                    f1t.append(f1_score(y_true, y_pred, average='macro'))
                    
                    accurracyt.append(accuracy_score(y_true, y_pred)) 
                    
                    if(len(set(y_pred)) != 1):
                        silhouettet.append(silhouette_score(dataset, y_pred))
                    else:
                        silhouettet.append(0)
                        
                                        
                    if(len(set(y_pred)) != 1):
                        davisbouldint.append(functions.davies_bouldin(dataset, y_pred, x.bestIndividual,y_true))
                    else:
                        davisbouldint.append(0)
                     
                                        
                    dunnt.append(functions.dunn(y_pred , euclidean_distances(dataset)))
                    

                Kbest = methods.cotovelo(otimo)      
                
                print(otimo)
                plt.plot(otimo)
                plt.show
                
                if (davisbouldint[Kbest-2] == 0):
                    conthbt = conthbt-1
                    contsilh= contsilh-1
                    contdb = contdb-1
                
                cft1 += cft[Kbest-2]
                fitness1 += otimo[Kbest-2]
                hbt1 += hbt[Kbest-2]
                arit1 += arit[Kbest-2]
                amit1 += amit[Kbest-2]
                f1t1 += f1t[Kbest-2]
                accurracyt1 += accurracyt[Kbest-2]
                silhouettet1 += silhouettet[Kbest-2]
                davisbouldint1 += davisbouldint[Kbest-2]
                dunnt1 +=dunnt[Kbest-2]
                Kmedio += Kbest
                
                
#cft1 += cft1/NumOfRuns
fitness1 = fitness1/NumOfRuns
hbt1 = hbt1/conthbt
arit1 = arit1/NumOfRuns
amit1 = amit1/NumOfRuns
f1t1 = f1t1/NumOfRuns
accurracyt1 = accurracyt1/NumOfRuns
silhouettet1 = silhouettet1/contsilh
davisbouldint1 = davisbouldint1/contdb
dunnt1 = dunnt1/NumOfRuns
Kmedio = Kmedio/NumOfRuns

print("Média das corridas")
print(cft1)
print("Fitness(Euclidiana): ", fitness1)
print("Calinski: ", hbt1)
print("ARI: ", arit1)
print("AMI: ", amit1)
print("F-measure: ", f1t1)
print("Acurácia: ", accurracyt1)
print("Silhueta: ", silhouettet1)
print("Davis-Boudin: ", davisbouldint1)
print("Dunn: ", dunnt1)
print("K médio: ", Kmedio)


#if (Flag==False):
#    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        
