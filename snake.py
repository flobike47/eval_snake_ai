import random
import itertools
import numpy
from NeuralNetwork import *

nbFeatures = 8
nbActions = 4

class Game:
    def __init__(self, hauteur, largeur):
        self.grille = [[0]*hauteur  for _ in range(largeur)]
        self.hauteur, self.largeur = hauteur, largeur
        self.serpent = [[largeur//2-i-1, hauteur//2] for i in range(4)]
        for (x,y) in self.serpent: self.grille[x][y] = 1
        self.direction = 3
        self.accessibles = [[x,y] for (x,y) in list(itertools.product(range(largeur), range(hauteur))) if [x,y] not in self.serpent]
        self.fruit = [0,0]
        self.setFruit()
        self.enCours = True
        self.steps = 0
        self.score = 4
    
    def setFruit(self):
        if (len(self.accessibles)==0): return
        self.fruit = self.accessibles[random.randint(0, len(self.accessibles)-1)][:]
        self.grille[self.fruit[0]][self.fruit[1]] = 2

    def refresh(self):
        nextStep = self.serpent[0][:]
        match self.direction:
            case 0: nextStep[1]-=1
            case 1: nextStep[1]+=1
            case 2: nextStep[0]-=1
            case 3: nextStep[0]+=1

        if nextStep not in self.accessibles:
            self.enCours = False
            return
        self.accessibles.remove(nextStep)
        if self.grille[nextStep[0]][nextStep[1]]==2:
            self.setFruit()
            self.steps = 0
            self.score+=1
        else:
            self.steps+=1
            self.grille[self.serpent[-1][0]][self.serpent[-1][1]] = 0
            self.accessibles.append(self.serpent[-1][:])
            self.serpent = self.serpent[:-1]
            if self.steps>self.hauteur*self.largeur:
                self.enCours = False
                return

        self.grille[nextStep[0]][nextStep[1]] = 1
        self.serpent = [nextStep]+self.serpent

    def getFeatures(self):
        # Initialise le vecteur de caractéristiques avec des zéros
        features = numpy.zeros(8)

        # Coordonnées de la tête du serpent
        head_x, head_y = self.serpent[0]

        # Points adjacents à la tête
        point_dessus = [head_x, head_y - 1]
        point_dessous = [head_x, head_y + 1]
        point_gauche = [head_x - 1, head_y]
        point_droite = [head_x + 1, head_y]

        # --- Caractéristiques 1-4: Détection des obstacles ---
        # [cite: 32] Obstacle au-dessus (mur ou corps)
        features[0] = 1 if head_y == 0 or self.grille[point_dessus[0]][point_dessus[1]] == 1 else 0
        # [cite: 33] Obstacle en dessous (mur ou corps)
        features[1] = 1 if head_y == self.hauteur - 1 or self.grille[point_dessous[0]][point_dessous[1]] == 1 else 0
        # [cite: 34] Obstacle à gauche (mur ou corps)
        features[2] = 1 if head_x == 0 or self.grille[point_gauche[0]][point_gauche[1]] == 1 else 0
        # [cite: 35] Obstacle à droite (mur ou corps)
        features[3] = 1 if head_x == self.largeur - 1 or self.grille[point_droite[0]][point_droite[1]] == 1 else 0

        # --- Caractéristiques 5-6: Position relative du fruit ---
        # [cite: 36] Vertical: 1 (au-dessus), -1 (en dessous), 0 (même ligne)
        features[4] = numpy.sign(head_y - self.fruit[1])
        # [cite: 37] Horizontal: 1 (à droite), -1 (à gauche), 0 (même colonne)
        features[5] = numpy.sign(self.fruit[0] - head_x)

        # --- Caractéristique 7: Direction actuelle ---
        # [cite: 38] 0: haut, 1: bas, 2: gauche, 3: droite
        features[6] = self.direction

        # --- Caractéristique 8: Distance normalisée au mur ---
        # [cite: 39, 40] Calcule la distance au mur dans la direction actuelle et la normalise
        distance = 0
        if self.direction == 0:  # Haut
            distance = head_y
            features[7] = distance / self.hauteur
        elif self.direction == 1:  # Bas
            distance = self.hauteur - 1 - head_y
            features[7] = distance / self.hauteur
        elif self.direction == 2:  # Gauche
            distance = head_x
            features[7] = distance / self.largeur
        elif self.direction == 3:  # Droite
            distance = self.largeur - 1 - head_x
            features[7] = distance / self.largeur

        return features
    
    def print(self):
        print("".join(["="]*(self.largeur+2)))
        for ligne in range(self.hauteur):
            chaine = ["="]
            for colonne in range(self.largeur):
                if self.grille[colonne][ligne]==1: chaine.append("#")
                elif self.grille[colonne][ligne]==2: chaine.append("F")
                else: chaine.append(" ")
            chaine.append("=")
            print("".join(chaine))
        print("".join(["="]*(self.largeur+2))+"\n")

