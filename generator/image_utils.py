import os
from skimage import color
from skimage import io
from copy import copy, deepcopy
import numpy as np



def generate_image_sequence(patterntensor, seq_dic, config, symbols, inner_symbols):
    image_sequence=[]
    for i in range(0, config['number_of_diagrams']):
        image_sequence.append(generate_single_image(i, patterntensor, seq_dic, config, symbols, inner_symbols))

    return image_sequence

def generate_single_image(frame, patterntensor, seq_dic, config, symbols, inner_symbols):
    pattern=patterntensor[:,:,frame]
    single_image = np.ones((int(config['height_and_width']), int(config['height_and_width'])))
    for x in range(0,len(pattern)):
        for y in range(0, len(pattern[0])):
            if pattern[x][y]>0:
                single_image=add_symbol(single_image,frame,x,y, pattern[x][y], seq_dic, config, symbols, inner_symbols)
    return single_image

def add_symbol(single_image,frame,x,y, sequence_number, seq_dic, config, symbols, inner_symbols):
    lower_x=int(config['distance_from_border'])+x*int(config['distance_between_symbols'])+x*int(config['symbol_size'])
    upper_x=lower_x+int(config['symbol_size'])
    lower_y=int(config['distance_from_border'])+y*int(config['distance_between_symbols'])+y*int(config['symbol_size'])
    upper_y=lower_y+int(config['symbol_size'])

    current_symbol=symbols[seq_dic[sequence_number].symbols[frame]]
    current_inner_symbol=inner_symbols[seq_dic[sequence_number].inner_symbols[frame]]
    merged_symbol=merge_symbol_with_inner_symbol(current_symbol, current_inner_symbol)
    single_image[lower_x:upper_x, lower_y:upper_y]=merged_symbol

    return single_image

def image_to_matrix(image_path):
    return color.rgb2gray(io.imread(image_path))

def merge_symbol_with_inner_symbol(symbol, inner_symbol):
    flag=False
    new_symbol = deepcopy(symbol)
    for x in range(0,len(symbol)):
        flag=False
        for y in range(0,len(symbol[0])):
            if symbol[x][y]<0.01:
                flag=True
            elif flag:
                if symbol[x][y]>0.01:
                    if inner_symbol[x][y]<0.01:
                        new_symbol[x][y]=inner_symbol[x][y]


    #clean up from the right
    for x in range(len(symbol)-1, -1, -1):
        for y in range(len(symbol)-1, -1, -1):
            if symbol[x][y]>0.99:
                new_symbol[x][y]=1.0
            else:
                break

    return new_symbol