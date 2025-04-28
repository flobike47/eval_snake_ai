import copy

import numpy
from NeuralNetwork import *
from snake import *

def eval(sol, gameParams):
    N = gameParams["nbGames"]
    H = gameParams["height"]
    W = gameParams["width"]

    total_score_sum = 0

    for i in range(N):
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
    for i in range(taillePopulation):
        nn = NeuralNetwork((arch[0],))
        for j in range(1, len(arch)):
            nn.addLayer(arch[j], "elu")
        population.append(Individu(nn))

    for sol in population: eval(sol, gameParams)
    population.sort(reverse=True, key=lambda sol:sol.score)
    
    return population

def optimize(taillePopulation, tailleSelection, pc, mr, arch, gameParams, nbIterations, nbThreads, scoreMax):
    # Initialisation de la population (déjà triée par score décroissant)
    population = initialization(taillePopulation, arch, gameParams)
    print(f"Génération 0: Meilleur score = {population[0].score:.4f}")

    for generation in range(nbIterations):
        # 1. Sélection des meilleurs individus
        selection = population[:tailleSelection]

        # 2. Création des enfants par croisement et mutation
        enfants = []
        while len(enfants) < taillePopulation - tailleSelection:
            # Choisir deux parents aléatoirement parmi la sélection
            idx1, idx2 = random.sample(range(tailleSelection), 2)
            parent1 = selection[idx1]
            parent2 = selection[idx2]

            # Croisement pour obtenir deux enfants
            nouveaux_enfants = crossover(parent1, parent2, pc, arch)

            # Mutation et évaluation des enfants
            for enfant in nouveaux_enfants:
                if len(enfants) < taillePopulation - tailleSelection:
                    mutation(enfant, mr)
                    eval(enfant, gameParams) # Évaluer le nouvel enfant
                    enfants.append(enfant)
                else:
                    break # Arrêter si on a assez d'enfants

        # 3. Nouvelle population = Sélection + Enfants
        population = selection + enfants

        # 4. Trier la nouvelle population par score
        population.sort(reverse=True, key=lambda sol:sol.score)

        # Affichage (optionnel)
        print(f"Génération {generation + 1}: Meilleur score = {population[0].score:.4f}")

        # Condition d'arrêt si le score max est atteint (optionnel)
        if population[0].score >= scoreMax:
            print("Score maximum atteint !")
            break

    # Retourner le réseau de neurones du meilleur individu trouvé
    return population[0].nn

def crossover(p1, p2, pc, arch):
    # Crée les structures pour les enfants (nouveaux réseaux)
    enfant1 = Individu(NeuralNetwork((arch[0],)))
    enfant2 = Individu(NeuralNetwork((arch[0],)))
    for j in range(1, len(arch)):
        enfant1.nn.addLayer(arch[j], "elu") # Assumant "elu" comme dans initialization
        enfant2.nn.addLayer(arch[j], "elu")

    # Si un nombre aléatoire > pc, les enfants sont des clones des parents
    if random.random() > pc:
        enfant1.nn = copy.deepcopy(p1.nn) # Copie profonde pour éviter les références partagées
        enfant2.nn = copy.deepcopy(p2.nn)
        return [enfant1, enfant2]

    # Sinon, appliquer le croisement pondéré par couche
    for i in range(len(p1.nn.layers)): # Parcourt chaque couche Dense
        layer_p1 = p1.nn.layers[i]
        layer_p2 = p2.nn.layers[i]
        layer_c1 = enfant1.nn.layers[i]
        layer_c2 = enfant2.nn.layers[i]

        alpha = random.random() # Coefficient alpha pour la couche

        # Croisement des poids
        layer_c1.weights = alpha * layer_p1.weights + (1 - alpha) * layer_p2.weights
        layer_c2.weights = (1 - alpha) * layer_p1.weights + alpha * layer_p2.weights

        # Croisement des biais
        layer_c1.bias = alpha * layer_p1.bias + (1 - alpha) * layer_p2.bias
        layer_c2.bias = (1 - alpha) * layer_p1.bias + alpha * layer_p2.bias

    return [enfant1, enfant2]

def mutation(enfant, mr):
    # Facteur d'amplitude pour la mutation (à ajuster si besoin)
    mutation_strength = 0.1 # Petit facteur pour ne pas trop déstabiliser

    for i in range(len(enfant.nn.layers)):
        layer = enfant.nn.layers[i]

        # Calcul des probabilités de mutation pour la couche
        prev_layer_size = layer.inputShape[0]
        layer_size = layer.outputShape[0]

        pm_biais = mr / layer_size if layer_size > 0 else 0
        pm_poids = mr / prev_layer_size if prev_layer_size > 0 else 0

        # Mutation des biais
        for j in range(layer_size):
            if random.random() < pm_biais:
                layer.bias[j] += random.gauss(0, mutation_strength) # Ajoute bruit Gaussien

        # Mutation des poids
        for r in range(prev_layer_size):
            for c in range(layer_size):
                if random.random() < pm_poids:
                    layer.weights[r, c] += random.gauss(0, mutation_strength) # Ajoute bruit Gaussien


