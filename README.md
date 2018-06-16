# Generic Module for Neuroevolution
This repository hold a generic module(class) that can be used for Neuroevolution (Genetic Algorithms) in various environments. It's heavily based on my [String Matcher](https://github.com/gurupunskill/string-matcher) implementation but instead of strings, we use Neural Networks.

The class is present [here](generic_genetic.py)

## Requirements
1. Numpy    : For everything
2. Pandas   : Just in case I want to use it to display stuff later
3. Keras    : Neural Networks and whatnot

## Usage
The class does not simulate the environment itself. `init_generation` creates a population of individuals depending on input parameters and returns the population. We need to simulate every individual manually and record their rewards. Then pass the rewards array into the `next_generation`.  
This is done to keep the module reusable.

### Initialization Parameters

```python
    trainer = neuroevolution(
        population_size = 55, 
        layers = [4, 8, 2], 
        mutation_rate = 0.20, 
        n_best_survivors = 5, 
        n_total_survivors = 10, 
        crossover_rate = 0.50
    )
```

1. `population_size`: size of the population
2. `layers`: the number of nodes in every layer of the neural network.  
    Expects an array of the form `[observation_dims, layer_1, layer_2, ..., action_dims]`
    `observation_dims` is the dimensions of the observation returned by the environment.
    `action_dim` is the number of possible actions the individual can take in the environment.
3. `mutation_rate`: the mutation rate
4. `crossover_rate`: the crossover rate (How likely it is to take parameters from the second parent). Generally 0.5.
5. `n_best_survivors`: number of best individuals in the population to be bred.
6. `n_total_survivors`: number of total individuals in the population to be bred.

### Initialize Generation
```python
    population = trainer.init_generation()
```
`init_generation` creates a keras dense neural network with softmax layers with units size provided by the layers tournament. It returns a ndarray of Keras models (which is the population of individuals).

### Next Generation
```python
    population = trainer.next_generation(fitness)
```
`next_generation` expects a fitness array, which should hold the respective rewards for every individual. `fitness[i]` should hold the reward for `current_population[i]`. `next_generation` creates the next population by crossing individuals based on the `mutation_rate` and the `crossover_rate`. The child will take weights from the first parent or from the second parent.


## I used this
I built a model for OpenAi gym's cartpole using this. Check it out [here](https://github.com/gurupunskill/cart-pole-genes/blob/master/cartpole.py)
