import numpy as np
import random
import json
from copy import copy, deepcopy

def fill_with_moving_patterns(patternmatrix, number_of_max_movements, movements):
    number_of_movements=random.randint(0,number_of_max_movements)
    for i in range(0,number_of_movements):
        patternmatrix, _=fill_with_single_moving_pattern(patternmatrix, movements)
    return patternmatrix

def fill_with_single_moving_pattern(patternmatrix, movements):
    counter=0
    while counter<len(movements):
        movement = random.sample(movements,1)[0]
        patternmatrix, is_success = apply_pattern(patternmatrix, movement)
        if is_success:
            return patternmatrix, True
        counter+=1

    return patternmatrix, False

def apply_pattern(patternmatrix, movement):
    x_max=len(patternmatrix)
    y_max=len(patternmatrix[0])
    all_possible_patterns=[]
    for x in range(0,x_max):
        for y in range(0,y_max):
            possible_pattern, is_possible=walkthrough(patternmatrix, movement, x, y)
            if is_possible:
                all_possible_patterns.append(possible_pattern)
    if len(all_possible_patterns)==0:
        return patternmatrix, False
    else:
        return random.sample(all_possible_patterns,1)[0], True

def walkthrough(patternmatrix, movement, x_start, y_start):
    sequence_len=len(movement['x'])
    x_max=len(patternmatrix)
    y_max=len(patternmatrix[0])
    patternnumber=np.amax(patternmatrix)+1

    #Drehen?
    if (random.random() < 0.5):
        temp=movement['x']
        movement['x']=movement['y']
        movement['y']=temp
    #*(-1) pro Zeile?
    if (random.random() < 0.5):
        movement['x']=np.array(movement['x'])*(-1)
    if (random.random() < 0.5):
        movement['y']=np.array(movement['y'])*(-1)

    new_matrix=deepcopy(patternmatrix)

    x_pos=x_start
    y_pos=y_start
    if new_matrix[x_pos][y_pos][0] == 0:
        new_matrix[x_pos][y_pos][0] = patternnumber
    else:
        return patternmatrix, False

    for current in range(0, sequence_len):
        x_pos = (x_pos + movement['x'][current]) % x_max
        y_pos = (y_pos + movement['y'][current]) % y_max
        if new_matrix[x_pos][y_pos][current+1]==0:
            new_matrix[x_pos][y_pos][current+1]=patternnumber
        else:
            return patternmatrix, False

    return new_matrix, True




