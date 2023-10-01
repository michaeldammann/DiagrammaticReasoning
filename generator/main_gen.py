import numpy as np
import json
from movement_utils import fill_with_moving_patterns
from still_utils import fill_with_still_patterns
from seq_utils import seqnumber_to_sequence_dict
from image_utils import image_to_matrix, generate_image_sequence
from os import listdir, makedirs
from os.path import isfile, join, exists
import argparse
from matplotlib import pyplot as plt


def main(savepath, n_seqs, number_of_max_movements, chance_per_still_spot):
    # read config
    with open('config/config.json') as f:
        config = json.load(f)

    # read movements
    with open('movements/movements.json') as f:
        movements = json.load(f)

    # read sequences
    with open('sequences/sequences.json') as f:
        sequences = json.load(f)

    # read symbols
    symbols = [f for f in listdir('symbols/') if isfile(join('symbols/', f))]
    symbols = [image_to_matrix('symbols/' + path) for path in symbols]

    # read inner symbols
    inner_symbols = [f for f in listdir('inner_symbols/') if isfile(join('inner_symbols/', f))]
    inner_symbols = [image_to_matrix('inner_symbols/' + path) for path in inner_symbols]

    if not exists(savepath):
        makedirs(savepath)

    for i in range(n_seqs):
        seqsavepath = join(savepath, str(i))
        if not exists(seqsavepath):
            makedirs(seqsavepath)
        # init pattern matrix with 0s
        patterntensor = np.zeros((4, 4, 6), dtype='int32')

        patterntensor = fill_with_moving_patterns(patterntensor, number_of_max_movements, movements)

        patterntensor = fill_with_still_patterns(patterntensor, chance_per_still_spot)

        # fill abstract pattern with actual images
        seq_dic = seqnumber_to_sequence_dict(patterntensor, sequences, symbols, inner_symbols)
        result = generate_image_sequence(patterntensor, seq_dic, config, symbols, inner_symbols)

        for img_idx in range(6):
            plt.imsave(join(seqsavepath, f'{img_idx}.png'), np.array(result)[img_idx], cmap=plt.get_cmap('gray'), vmin=0, vmax=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath', required=True, type=str)
    parser.add_argument('--n_seqs', required=True, type=int)
    parser.add_argument('--number_of_max_movements', required=False, default=5, type=int)
    parser.add_argument('--chance_per_still_spot', required=False, default=0.5, type=int)
    args = parser.parse_args()
    main(args.savepath, args.n_seqs, args.number_of_max_movements, args.chance_per_still_spot)
