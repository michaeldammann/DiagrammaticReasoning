import numpy as np
import random

def fill_with_still_patterns(patternmatrix, chance_per_spot):
    free_spots=determine_free_spots(patternmatrix)
    for i in range(0,len(free_spots)):
        if (random.random() < chance_per_spot):
            patternmatrix=fill_single_free_spot(patternmatrix,free_spots[i])

    return patternmatrix


def determine_free_spots(patternmatrix):
    results=[]
    for x in range(0,len(patternmatrix)):
        for y in range(0, len(patternmatrix[0])):
            is_possible=True
            for z in range(0,len(patternmatrix[0][0])):
                if patternmatrix[x][y][z]>0:
                    is_possible=False
                    break
            if is_possible:
                results.append([x,y])

    return results

def fill_single_free_spot(patternmatrix, coordinates):
    x=coordinates[0]
    y=coordinates[1]
    patternnumber=np.amax(patternmatrix)+1
    for z in range(0,len(patternmatrix[x][y])):
        patternmatrix[x][y][z]=patternnumber

    return patternmatrix

