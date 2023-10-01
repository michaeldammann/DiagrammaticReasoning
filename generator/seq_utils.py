import numpy as np
import random
from Sequence import Sequence

def seqnumber_to_sequence_dict(patternmatrix, sequences, symbols, inner_symbols):
    seqnumbers=[]
    for x in range(0, len(patternmatrix)):
        for y in range(0, len(patternmatrix[0])):
            for z in range(0, len(patternmatrix[0][0])):
                if patternmatrix[x][y][z] not in seqnumbers:
                    seqnumbers.append(patternmatrix[x][y][z])

    seq_dic = {seqnumber: Sequence(random.randint(0,len(np.asarray(sequences))-1),random.randint(0,len(np.asarray(sequences))-1)) for seqnumber in seqnumbers}

    for k in seq_dic:
        seq_dic[k].symbols=determine_symbols(seq_dic[k].seq_type, sequences, symbols)
        seq_dic[k].inner_symbols=determine_symbols(seq_dic[k].seq_inner_type, sequences, inner_symbols)

    return seq_dic

def determine_symbols(seq_type, sequences, symbols):
    result=[]

    sequence=sequences[seq_type]
    number_of_symbols=find_number_of_symbols(sequence)

    used_symbols=[]
    random_symbols=[]
    for i in range(0, number_of_symbols):
        sampled_symbol=random.randint(0,len(symbols)-1)
        while sampled_symbol in used_symbols:
            sampled_symbol = random.randint(0, len(symbols) - 1)

        used_symbols.append(sampled_symbol)

    for j in range(0, len(symbols)):
        if j not in used_symbols:
            random_symbols.append(j)

    for elem in sequence:
        if elem=='r':
            result.append(random.sample(random_symbols,1)[0])
        else:
            result.append(used_symbols[elem-1])

    return result



def find_number_of_symbols(sequence):
    new_array=[]
    for elem in sequence:
        if elem!='r':
            new_array.append(elem)
    return max(new_array)

