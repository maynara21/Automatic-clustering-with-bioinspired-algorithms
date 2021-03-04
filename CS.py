import math
import numpy
import random
import time
from solution import solution


    
def get_cuckoos(nest,best,lb,ub,n,dim,clusters):
    
    # perform Levy flights
    tempnest=numpy.zeros((n,clusters,dim))
    tempnest=numpy.array(nest)
    beta=3/2
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    s=numpy.zeros((clusters,dim))
    
    for i in range (0,n):
        
        s=nest[i,:,:]
        step=numpy.zeros((clusters,dim))
        stepsize=numpy.zeros((clusters,dim))
        u=numpy.random.randn(clusters,dim)*sigma
        v=numpy.random.randn(clusters,dim)
        
        for j in range (0,clusters):
            for k in range (0,dim):
            
                step[j,k]=u[j,k]/abs(v[j,k])**(1/beta)
                stepsize[j,k]=0.01*(step[j,k]*(s[j,k]-best[j,k]))
                s[j,k] = s[j,k]+stepsize[j,k]*numpy.random.random()
                tempnest[i,j,k]=numpy.clip(s[j,k], lb[k], ub[k])

    return tempnest


def get_best_nest(nest,newnest,fitness,n,dim,objf,data,dataset,clusters):
    
    #Evaluating all new solutions
    tempnest=numpy.zeros((n,clusters,dim))
    tempnest=numpy.copy(nest)
    
    for j in range(0,n):
  
        fnew=objf(newnest[j,:,:],data,dataset,clusters)
        if fnew<=fitness[j]:
           fitness[j]=fnew
           tempnest[j,:,:]=newnest[j,:,:]
        
    # Find the current best
    fmin = min(fitness)
    K=numpy.argmin(fitness)
    bestlocal=tempnest[K,:,:]

    return fmin,bestlocal,tempnest,fitness


# Replace some nests by constructing new solutions/nests
def empty_nests(nest,pa,n,dim,clusters):

    # Discovered or not 
    tempnest=numpy.zeros((n,clusters,dim))

    K=numpy.random.uniform(0,1,(n,clusters,dim))>pa
    
    stepsize=random.random()*(nest[numpy.random.permutation(n),:,:]-nest[numpy.random.permutation(n),:,:])

    tempnest=nest+stepsize*K
 
    return tempnest

##########################################################################################################


def CS(objf,lb,ub,dim,n,N_IterTotal,data,dataset,clusters):

    # Discovery rate of alien eggs/solutions
    pa=0.25

    # Initialize nests randomely
    nest=numpy.random.rand(n,clusters,dim)
    
    for i in range(dim):
        nest[:,:,i] *= (ub[i]-lb[i])+lb[i]
       
    
    new_nest=numpy.zeros((n,clusters,dim))
    new_nest=numpy.copy(nest)
    
    bestnest=numpy.zeros((clusters,dim))
             
     
    fitness=numpy.zeros(n) 
    fitness.fill(float("inf"))

    s=solution()

     
    print("CS is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    fmin,bestnest,nest,fitness =get_best_nest(nest,new_nest,fitness,n,dim,objf,data,dataset,clusters)
    convergence=numpy.zeros(N_IterTotal)
    
    # Main loop counter
    for iter in range (0,N_IterTotal):
        
        # Generate new solutions (but keep the current best)
         new_nest=get_cuckoos(nest,bestnest,lb,ub,n,dim,clusters)
         
         
         # Evaluate new solutions and find best
         fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim,objf,data,dataset,clusters)
         
        
         new_nest=empty_nests(new_nest,pa,n,dim,clusters)
         
        
        # Evaluate new solutions and find best
         fnew,best,nest,fitness=get_best_nest(nest,new_nest,fitness,n,dim,objf,data,dataset,clusters)
    
         if fnew<fmin:
            fmin=fnew
            bestnest=best
    
         if (iter%1==0):
            print(['At iteration '+ str(iter)+ ' the best fitness is '+ str(fmin)]);
         convergence[iter]=fmin

    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    s.optimizer="PSO"
    s.objfname=objf.__name__
    s.bestIndividual = bestnest
    s.last = fmin
    #print(bestnest)
    
    return s
    

