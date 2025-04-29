import copy

import numpy
from NeuralNetwork import *
from snake import *

def eval(sol, gameParams):
    N = gameParams["nbGames"]
    H = gameParams["height"]
    W = gameParams["width"]
    
    total_score_sum = 0

    for _ in range(N):
        game = Game(H, W)
        initial_snake_length = len(game.serpent)
        while game.enCours:
            features = game.getFeatures()
            predicted_direction = sol.nn.predict(features)
            game.direction = predicted_direction
            game.refresh()
        p_i = game.score - initial_snake_length
        s_i = game.steps
        total_score_sum += (1000 * p_i + s_i)

    final_score = total_score_sum / (N * H * W * 1000)
    sol.score = final_score
    return final_score

'''
Représente une solution avec
_un réseau de neurones
_un score (à maximiser)

vous pouvez ajouter des attributs ou méthodes si besoin
'''
class Individu:
    def __init__(self, nn):
        self.nn = nn
        self.score = 0


'''
La méthode d'initialisation de la population est donnée :
_on génère N individus contenant chacun un réseau de neurones (de même format)
_on évalue et on trie des individus
'''
def initialization(taillePopulation, arch, gameParams):
    population = []
    for _ in range(taillePopulation):
        nn = NeuralNetwork((arch[0],))
        for j in range(1, len(arch)):
            nn.addLayer(arch[j], "elu")
        population.append(Individu(nn))

    for sol in population: eval(sol, gameParams)
    population.sort(reverse=True, key=lambda sol:sol.score)
    
    return population

def optimize(taillePopulation, tailleSelection, pc, mr, arch, gameParams, nbIterations, nbThreads, scoreMax):
    population = initialization(taillePopulation, arch, gameParams)
    print(f"Génération 0: Meilleur score = {population[0].score:.4f}")

    for generation in range(nbIterations):
        selection = population[:tailleSelection]
        enfants = []
        while len(enfants) < taillePopulation - tailleSelection:
            idx1, idx2 = random.sample(range(tailleSelection), 2)
            parent1 = selection[idx1]
            parent2 = selection[idx2]
            nouveaux_enfants = crossover(parent1, parent2, pc, arch)
            for enfant in nouveaux_enfants:
                if len(enfants) < taillePopulation - tailleSelection:
                    mutation(enfant, mr)
                    eval(enfant, gameParams)
                    enfants.append(enfant)
                else:
                    break
        population = selection + enfants
        population.sort(reverse=True, key=lambda sol:sol.score)
        print(f"Génération {generation + 1}: Meilleur score = {population[0].score:.4f}")
        if population[0].score >= scoreMax:
            print("Score maximum atteint !")
            break
    return population[0].nn

def crossover(p1, p2, pc, arch):
    enfant1 = Individu(NeuralNetwork((arch[0],)))
    enfant2 = Individu(NeuralNetwork((arch[0],)))
    for j in range(1, len(arch)):
        enfant1.nn.addLayer(arch[j], "elu")
        enfant2.nn.addLayer(arch[j], "elu")
    if random.random() > pc:
        enfant1.nn = copy.deepcopy(p1.nn)
        enfant2.nn = copy.deepcopy(p2.nn)
        return [enfant1, enfant2]
    for i in range(len(p1.nn.layers)):
        layer_p1 = p1.nn.layers[i]
        layer_p2 = p2.nn.layers[i]
        layer_c1 = enfant1.nn.layers[i]
        layer_c2 = enfant2.nn.layers[i]
        alpha = random.random()
        layer_c1.weights = alpha * layer_p1.weights + (1 - alpha) * layer_p2.weights
        layer_c2.weights = (1 - alpha) * layer_p1.weights + alpha * layer_p2.weights
        layer_c1.bias = alpha * layer_p1.bias + (1 - alpha) * layer_p2.bias
        layer_c2.bias = (1 - alpha) * layer_p1.bias + alpha * layer_p2.bias
    return [enfant1, enfant2]

def mutation(enfant, mr):
    mutation_strength = 0.1

    for i in range(len(enfant.nn.layers)):
        layer = enfant.nn.layers[i]
        prev_layer_size = layer.inputShape[0]
        layer_size = layer.outputShape[0]
        pm_biais = mr / layer_size if layer_size > 0 else 0
        pm_poids = mr / prev_layer_size if prev_layer_size > 0 else 0
        for j in range(layer_size):
            if random.random() < pm_biais:
                layer.bias[j] += random.gauss(0, mutation_strength)
        for r in range(prev_layer_size):
            for c in range(layer_size):
                if random.random() < pm_poids:
                    layer.weights[r, c] += random.gauss(0, mutation_strength)


