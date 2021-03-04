import random
import numpy
import math
from solution import solution
import time


def WOA(objf,lb,ub,dim,SearchAgents_no,Max_iter,data,dataset,clusters):
    
    # initialize position vector and score for the leader
    Leader_pos=numpy.zeros((clusters,dim))
    Leader_score=float("inf")  
    
    
    #Initialize the positions of search agents
    Positions=numpy.random.uniform(0,1,(SearchAgents_no,clusters, dim))
    
    for i in range(dim):
        Positions[:,:,i] *= (ub[i]-lb[i])+lb[i]
        
    #Initialize convergence
    convergence_curve=numpy.zeros(Max_iter)
    
    s=solution()

    print("WOA is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    t=0  # Loop counter
   
    A = numpy.zeros((clusters,dim))
    C = numpy.zeros((clusters,dim))
    
    D_X_rand=numpy.zeros((clusters,dim))
    D_Leader=numpy.zeros((clusters,dim))
    distance2Leader=numpy.zeros((clusters,dim))
    
    print(clusters)
    print(dim)
    
    # Main loop
    while t<Max_iter:
       
        
        for i in range(0,SearchAgents_no):
            for j in range(0,clusters):
                for k in range(0,dim):
            # Return back the search agents that go beyond the boundaries of the search space
                    Positions[i,j,k]=numpy.clip(Positions[i,j,k], lb[k], ub[k])
            
            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:,:], data, dataset,clusters)
            # Update the leader
            if fitness<Leader_score: 
                Leader_score=fitness; 
                Leader_pos=Positions[i,:,:]
        
     #   a = numpy.zeros(clusters)
     #   a2 = numpy.zeros(clusters)
        
        a = 2-t*((2)/Max_iter); # a decreases linearly fron 2 to 0 in Eq. (2.3)
        
        # a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a2=-1+t*((-1)/Max_iter);
    
        
        
        
        # Update the Position of search agents 
        for i in range(0,SearchAgents_no):
            
            r1=numpy.random.randn(clusters,dim) # r1 is a random number in [0,1]
            r2=numpy.random.randn(clusters,dim) # r2 is a random number in [0,1]
            
            for j in range(0,clusters):
               for k in range(0,dim): 
                
                    #A[j,k]=2*random.random() -1  # Eq. (2.3) in the paper

                    A[j,k]=2*a*r1[j,k]-a  # Eq. (2.3) in the paper
                    C[j,k]=2*r2[j,k]      # Eq. (2.4) in the paper
            
            b=1;               #  parameters in Eq. (2.5)
            l=(a2-1)*random.random()+1   #  parameters in Eq. (2.5)    
            p = random.random()        # p in Eq. (2.6)
        
            rand_leader_index = math.floor(SearchAgents_no*random.random());
            X_rand = Positions[rand_leader_index,:,:]
            
            
            
            for j in range(0, clusters):
               for k in range(0,dim):
                    if p<0.5:
                                       
                        if abs(A[j,k])>=1:
                            
                            
                            D_X_rand[j,k]=abs(C[j,k]*X_rand[j,k]  -Positions[i,j,k]) 
                            Positions[i,j,k]=X_rand[j,k]-A[j,k]*D_X_rand[j,k]
                        
                        elif abs(A[j,k])<1:
                            D_Leader[j,k]=abs(C[j,k]*Leader_pos[j,k]-Positions[i,j,k]) 
                            Positions[i,j,k]=Leader_pos[j,k]-A[j,k]*D_Leader[j,k]
                                
                    elif p>=0.5:
                      
                        distance2Leader[j,k]=abs(Leader_pos[j,k]-Positions[i,j,k])
                        # Eq. (2.5)
                        Positions[i,j,k]=distance2Leader[j,k]*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j,k]
                    
                
        convergence_curve[t]=Leader_score
        if (t%1==0):
               print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Leader_score)]);
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="WOA"   
    s.objfname=objf.__name__
    s.bestIndividual = Leader_pos
    s.last = Leader_score

    return s


