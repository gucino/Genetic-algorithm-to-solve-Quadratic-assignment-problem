# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:01:15 2020

@author: Tisana
"""

import numpy as np
from matplotlib import pyplot as plt

#quadratic assignment problem
distance_matrix=np.array([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]])
flow_matrix=np.array([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]])
current_solution=np.array(["A","B","C","D","E","F","G","H"])

#hyper-parameter
num_population=50
prob_mutation=0.3
num_generation=5000

#cost function
def cost_function(distance,flow_matrix):
    cost=(distance*flow_matrix).sum()
    return cost
#dict btw  index and department name
convert={"A":0,
         "B":1,
         "C":2,
         "D":3,
         "E":4,
         "F":5,
         "G":6,
         "H":7}
#transform list of depeartment into distance matrix
def solution_function(solution):
    index_list=[]
    for each_letter in solution:
        index_list.append(convert[each_letter])
    new_solution= np.copy(distance_matrix)
    new_solution=new_solution[index_list,:]
    new_solution=new_solution[:,index_list]
    return new_solution
#generate new solution
def move(a):
    index_one=np.random.randint(0,len(a))
    index_two=np.random.randint(0,len(a))
    
    while index_one==index_two:
        index_two=np.random.randint(0,len(a))
    new_solution=np.copy(a)
    new_solution[index_one]=a[index_two]
    new_solution[index_two]=a[index_one]
    return new_solution
#generate iitial population
def initial_population_function(num_population):
    population_list=[]
    for i in range(0,num_population):
        population_list.append(move(current_solution))
    return population_list
#fitness value for each chromosome
def fitness_value_function(population_list):
    FV_list=[]
    for each_chromosome in population_list:
        distance=solution_function(each_chromosome)
        FV_list.append(1/cost_function(distance,flow_matrix))
    return FV_list
        
#selection function : roulette selection wheel
def selection_function(fitness_value,population_list):
    total=sum(fitness_value)
    prob_list=np.array(fitness_value)/total
    index_list=np.arange(0,len(fitness_value))
    index1,index2=np.random.choice(a=index_list,p=prob_list,size=2)
    FV_p1=fitness_value[index1]
    FV_p2=fitness_value[index2]
    return population_list[index1],population_list[index2],index1,index2,FV_p1,FV_p2

#cross over and feasibility check   
def cross_over_function(parent1,parent2):

    point1=np.random.randint(0,len(parent1))
    point2=np.random.randint(0,len(parent1))
    while point1==point2:
        point2=np.random.randint(0,len(parent1))
    
    #crossover
    a=min(point1,point2)
    b=max(point1,point2)
    children1=(parent1[:a]).tolist()+(parent2[a:b]).tolist()+(parent1[b:]).tolist()
    children2=(parent2[:a]).tolist()+(parent1[a:b]).tolist()+(parent2[b:]).tolist()
    
    #check feasibility
    ok=1
    for letter in convert:
        if letter not in children1 or letter not in children2:
            ok=0
    if ok==1:
        return children1,children2
    else:
        return parent1,parent2

#mutation
def mutation_function(children1,children2,prob_mutation):
    rand=np.random.random()
    if rand<prob_mutation:
        mutate_children1=move(children1)
    else:
        mutate_children1=children1
    rand=np.random.random()
    if rand<prob_mutation:
        mutate_children2=move(children2)    
    else:
        mutate_children2=children2
    return mutate_children1,mutate_children2
    
    
###################################################################
###################################################################
###################################################################
#start


#generate initial population
initial_population=initial_population_function(num_population)


#for each generation
avg_FV_per_gen=[]
best_FV_per_gen=[]
best_cost_per_gen=[]
for each_gen in range(0,num_generation):

    #fitness per gen
    fitness_value_list=fitness_value_function(initial_population)
    
    #collect information
    avg_FV_per_gen.append(np.mean(fitness_value_list))
    best_FV_per_gen.append(max(fitness_value_list))
    best_cost_per_gen.append(1/max(fitness_value_list))
    
    #selection process
    parent1,parent2,index_p1,index_p2,FV_p1,FV_p2=selection_function(fitness_value_list\
                                                         ,initial_population)
    
    #cross over
    children1,children2=cross_over_function(parent1,parent2)


    #mutation
    mutate_children1,mutate_children2=mutation_function(children1,children2,prob_mutation)
    
    #elitism
    dist_c1=solution_function(mutate_children1)
    FV_c1=1/cost_function(dist_c1,flow_matrix)

    dist_c2=solution_function(mutate_children2)
    FV_c2=1/cost_function(dist_c2,flow_matrix)
    
    if FV_c1>FV_p1:
        initial_population[index_p1]=np.array(mutate_children1)
    if FV_c2>FV_p2:
        initial_population[index_p2]=np.array(mutate_children2)

    print(each_gen," of ",num_generation," best cost : ",1/max(fitness_value_list))
#plot
plt.figure()
plt.title("avg FV per generation")
plt.ylabel("avg FV")
plt.xlabel("generation")
plt.plot(np.arange(0,len(avg_FV_per_gen)),avg_FV_per_gen)

plt.figure()
plt.title("best FV per generation")
plt.ylabel("best FV")
plt.xlabel("generation")
plt.plot(np.arange(0,len(best_FV_per_gen)),best_FV_per_gen)

plt.figure()
plt.title("best cost per generation")
plt.ylabel("best cost")
plt.xlabel("generation")
plt.plot(np.arange(0,len(best_cost_per_gen)),best_cost_per_gen)