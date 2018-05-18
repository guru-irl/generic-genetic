class neuroevolution:
    
    def __init__ (population_size, layers, mutation_rate, n_best_survivors, n_total_survivors, crossover_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_best_survivors = n_best_survivors
        self.n_total_survivors = n_total_survivors
        self.crossover_rate = crossover_rate
        self.best_models = []
        
    def init_generation():
        self.current_population = []
        for _ in range(0, self.population_size):
            model = Sequential()
            model.add(Dense(input_dim = self.layers[0], units = self.layers[1], activation='softmax'))
            for i in range(2, len(self.layers)):
                model.add(Dense(units = self.layers[i], activation='softmax'))
            current_population.append(model)
        return current_population
    
    def cross(parent_1, parent_2, crossover_rate, mutation_rate):
        weights1 = parent_1.get_weights()
        weights2 = parent_2.get_weights()
    
        child_weight_1 = deepcopy(weights1)
    
        for i in range(0, len(weights1),2):
            for j in range(0, len(weights1[i])):
                for k in range(0, len(weights1[i][j])):
                    if random.uniform(0,1) > crossover_rate:
                        child_weight_1[i][j][k] = weights2[i][j][k]
                        print(i, j, k)
            
                    if random.uniform(0,1) < mutation_rate:
                        child_weight_1[i][j][k] += random.uniform(-1, 1)
                
                
        child_1 = Sequential.from_config(parent_1.get_config())
        child_1.set_weights(child_weight_1)
    
        return child_1
    
    def next_generation(fitness):
        self.current_population = [individual for _,individual in sorted(zip(fitness, self.current_population))]
        survivors = self.current_population[0:self.n_best_individuals]
        self.best_models.append(self.current_population[0])
        for _ in range(0, self.n_total_survivors - self.n_best_individuals):
            survivors.append(random.randint(self.n_best_individuals, len(self.current_population)))
        new_generation = []
        for i in range(0, len(survivors)):
            for j in range(i, len(survivors)):
                if (i == j):
                    new_generation.append(survivors[i])
                else:
                    new_generation.append(cross(survivors[i], survivors[j], 0.5, 0.2))
        self.current_population = new_generation
        return current_population
    
    def history():
        return self.best_models