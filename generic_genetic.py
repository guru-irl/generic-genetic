import numpy as np 
import pandas as pd
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from copy import deepcopy

class neuroevolution:
    
    def __init__ (self, population_size, layers, mutation_rate, n_best_survivors, n_total_survivors, crossover_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_best_survivors = n_best_survivors
        self.n_total_survivors = n_total_survivors
        self.crossover_rate = crossover_rate
        self.layers = layers
        self.best_models = []
        
    def init_generation(self):
        self.current_population = np.asarray([])
        
        for _ in range(0, self.population_size):
            model = Sequential()
            model.add(
                Dense(
                    input_dim = self.layers[0], 
                    units = self.layers[1], 
                    activation='softmax', 
                    kernel_initializer='random_uniform', 
                    bias_initializer='random_uniform'
                )
            )

            for i in range(2, len(self.layers)):
                model.add(
                    Dense(
                        units = self.layers[i], 
                        activation='softmax', 
                        kernel_initializer='random_uniform', 
                        bias_initializer='random_uniform'
                    )
                )
                
            self.current_population = np.append(self.current_population, model)
        
        return self.current_population
    
    def cross(self, parent_1, parent_2):
        weights1 = parent_1.get_weights()
        weights2 = parent_2.get_weights()
    
        child_weight_1 = deepcopy(weights1)
    
        for i in range(0, len(weights1),2):
            for j in range(0, len(weights1[i])):
                for k in range(0, len(weights1[i][j])):
                    if random.uniform(0,1) > self.crossover_rate:
                        child_weight_1[i][j][k] = weights2[i][j][k]
                    if random.uniform(0,1) < self.mutation_rate:
                        child_weight_1[i][j][k] += random.uniform(-1, 1)
                     
        child_1 = Sequential.from_config(parent_1.get_config())
        child_1.set_weights(child_weight_1)
    
        return child_1
    
    def next_generation(self, fitness):
    
        self.current_population = self.current_population[fitness.argsort()[::-1]]
        survivors = self.current_population[0:self.n_best_survivors]
        self.best_models.append(self.current_population[0])
        
        for _ in range(0, self.n_total_survivors - self.n_best_survivors):
            survivors = np.append(survivors, self.current_population[random.randint(self.n_best_survivors, len(self.current_population) - 1)])
        new_generation = np.asarray([])
        
        for i in range(0, len(survivors)):
            for j in range(i, len(survivors)):
                if (i == j):
                    new_generation = np.append(new_generation, survivors[i])
                else:
                    new_generation = np.append(new_generation, self.cross(survivors[i], survivors[j]))
        
        self.current_population = new_generation
        return self.current_population
    
    def history(self):
        return self.best_models