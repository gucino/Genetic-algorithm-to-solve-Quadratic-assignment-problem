#purpose
This code aim to solve quadratic assignment problem using GA (Genetic Algorithm) algorithm

#problem description
There are 8 departments .
There is distance between each department as well as the flow.
The goal is to have the lowest cost (flow * distance) by rearanging the distance matrix 
given that the flow matrix is fixed
This problem can be further adapted to be travelling salesman problem

#detail of algorithm
Fitness value : the inverse of cost that is  1/(flow * distance) 
Selection methodn : Roulette wheel
Cross over : 2 point cross over
Mutation : randommly swap 2 genome

#reult
model achieved 214 cost
Results are shown in "avg_FV_per_generation", "cost_per_generation.png"
and "FV_per_generation.png"


