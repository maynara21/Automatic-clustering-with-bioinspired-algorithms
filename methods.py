import numpy
import math

def cotovelo (pontos):
    p1 = numpy.empty([2])
    p2 = numpy.empty([2])
    p3 = numpy.empty([2])
    
    p1 = [1,pontos[0]]
    p2 = [len(pontos), pontos[len(pontos)-1]]
    
    dist = 0
    eps = 0
    div = 0
    
    for i in range(0,len(pontos)):
        print(i)
        p3[0] = i
        p3[1] = pontos[i]
        
        d = (p2[1] - p1[1]) * p3[0]
        d = d - ((p2[0] - p1[0]) * p3[1])
        d = d + (p2[0] * p1[1])
        d = d - (p2[1] * p1[0])
        d = abs(d)
        
        div = (p2[1] - p1[1]) ** 2 
        div = div + (p2[0] - p1[0]) ** 2
        
        div = math.sqrt(div)
        
        d = d/div
        
        if (d>= dist):
            dist = d
            eps = p3[1]
            Kopt = i+2
    
    
    return Kopt