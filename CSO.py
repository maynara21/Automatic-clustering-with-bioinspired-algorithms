import numpy as num
import random
import time
from solution import solution


def seek_mode(seekSize,posVectorSeek,dim,objf,data,dataset,clusters,lb,ub):
    
    
    temp = num.zeros(shape = (5,clusters,dim))
    ftemp = num.zeros(5)
   
    for i in range(0,seekSize):
        
        for j in range(0,5):
            
            temp[j,:,:] = posVectorSeek[i,:,:]
            y = num.random.randint(0,clusters)
            z = num.random.randint(0,dim)
          #  for k in range(0,clusters):
            temp[j,y,z] = num.round(num.random.rand(),2)
            temp[j,y,z] = temp[j,y,z]*(ub[z]-lb[z])-lb[z]
                
            ftemp[j] = objf(temp[j,:,:],data,dataset,clusters)
                
    
        best_index = num.argmin(ftemp)
        posVectorSeek[i,:,:] = temp[best_index,:,:]
        temp = num.zeros(shape = (5,clusters,dim))
        ftemp = num.zeros(5)
        
    return posVectorSeek


def trace_mode(seekSize,catNumber,traceSize,posVectorTrace,velVectorTrace,dim,objf,data,dataset,clusters,lb,ub):
    
    
    ftrace = num.zeros(traceSize)
    rnd=num.zeros(shape = (clusters,dim))
    
    for i in range(0,traceSize):
        ftrace[i] = objf(posVectorTrace[i,:,:],data,dataset,clusters)
  
    best_index = num.argmin(ftrace)
    gbest = num.zeros(shape = (traceSize,clusters,dim))
    
    
    for i in range(0,traceSize):
        gbest[i,:,:] = posVectorTrace[best_index,:,:]
        
    for i in range(seekSize,catNumber):
        for j in range(0, clusters):
            for k in range(0, dim):
                rnd[j,k] =  num.round(num.random.rand(),2)
        velVectorTrace[i,:,:] = velVectorTrace[i,:,:] + rnd*2*(gbest[i-seekSize,:,:]-posVectorTrace[i-seekSize,:,:])
    
    posVectorTrace = posVectorTrace + velVectorTrace[seekSize:,:,:]
    
    
    for j in range(0,clusters):
        for k in range(0,dim):
            posVectorTrace[:,j,k] = num.clip( posVectorTrace[:,j,k], lb[k], ub[k])
    
    
    return velVectorTrace,posVectorTrace
    

def find_gen(catNumber,seekSize,traceSize,posVectorSeek,posVectorTrace,velVectorTrace,dim,objf,data,dataset,clusters,lb,ub):
    
    
    posVectorSeek = seek_mode(seekSize,posVectorSeek,dim,objf,data,dataset,clusters,lb,ub)
    velVectorTrace,posVectorTrace = trace_mode(seekSize,catNumber,traceSize,posVectorTrace,velVectorTrace,dim,objf,data,dataset,clusters,lb,ub)
    
    
    return posVectorSeek,posVectorTrace,velVectorTrace
    
    
def CSO(objf,lb,ub,dim,catNumber,iters,data,dataset,clusters):
    
   
    seekSize = int(catNumber*0.8)
    traceSize = catNumber - seekSize
    
    posVectorSeek = num.random.rand(seekSize,clusters,dim) 
    posVectorTrace = num.random.rand(traceSize,clusters,dim)
    velVectorTrace = num.random.rand(catNumber,clusters,dim)
    
    
    for i in range(dim):
        posVectorSeek[:,:,i] *= (ub[i]-lb[i])+lb[i]
        posVectorTrace[:,:,i] *= (ub[i]-lb[i])+lb[i]
    
    fitness = num.zeros(catNumber) 
    fmin = float("inf")
    

    convergence = num.zeros(iters)
    
    
    s = solution()
    
    print("CSO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    

    for iter in range(0,iters):
        
        
        posVectorSeek,posVectorTrace,velVectorTrace = find_gen(catNumber,seekSize,traceSize,posVectorSeek,posVectorTrace,velVectorTrace,dim,objf,data,dataset,clusters,lb,ub)
        
        c = num.concatenate((posVectorSeek,posVectorTrace),axis = 0)
        
        for j in range(0,catNumber):
            fitness[j]=objf(c[j,:,:],data,dataset,clusters)
        
        index = num.argmin(fitness)
        fnew = fitness[index]
        
        if(fnew<fmin):
            fmin = fnew
            bestcat = c[index,:,:]
            
       
        d = num.concatenate((c,velVectorTrace),axis=1)
        num.random.shuffle(d)
        
        e = d[:,:clusters,:]
        
        velVectorTrace = d[:,clusters:,:]
        posVectorSeek = e[:seekSize,:,:]
        posVectorTrace = e[seekSize:,:,:]
        
        
    
        if (iter%1==0):
            print(['At iteration '+ str(iter)+ ' the best fitness is '+ str(fmin)]);
        
        convergence[iter]=fmin
     
        
    c = num.concatenate((posVectorSeek,posVectorTrace),axis = 0)
        
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    s.optimizer="CSO"   
    s.objfname=objf.__name__
    s.bestIndividual = bestcat
    s.last = fmin
#    
    return s    
