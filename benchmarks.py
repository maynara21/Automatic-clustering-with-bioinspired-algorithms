import math

def distances(particle, data, dataset,clusters):
    somatoria = 0
    for i in range(data.namostras):
        melhor = float("inf")
        for cluster in range(clusters):
            distancia = 0
            for dim in range(data.ndim):
                distancia = distancia + pow (particle[cluster][dim] - dataset[i][dim],2)
            distancia = math.sqrt(distancia)
            if (distancia < melhor):
                melhor = distancia
        somatoria = somatoria + melhor
    return somatoria


def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {  0: ["distances"],
            }
    return param.get(a, "nothing")