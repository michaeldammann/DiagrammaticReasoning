from keras.layers import Dense, Add, BatchNormalization


def rel_net(object_combinations, g_layers, g_dim, f_layers, f_dim, out_dim):
    '''
    A Keras implementation of the RelNet by Santoro et al.: A simple neural network module for relational reasoning,
    NeurIPS 2017
    :param object_combinations: List of lists of concatenations of (o_i, o_j), where o_i and o_j are
    objects from the object set O = {o_1, o_2,...o_n}
    :param g_layers: Number of dense layers in g
    :param g_dim: Units per dense layer in the g module
    :param f_layers: Number of dense layers in f
    :param f_dim: Units per
    dense layer in the f module
    :param out_dim: Output dimension

    :return: Output of the Model
    '''

    # g module
    dense0 = Dense(g_dim, activation="relu")
    dense0outputs = []
    for j in range(0, len(object_combinations)):
        dense0outputs.append(dense0(object_combinations[j]))

    bn0 = BatchNormalization()
    outputs = []
    for j in range(0, len(object_combinations)):
        outputs.append(bn0(dense0outputs[j]))

    for _ in range(g_layers - 1):
        dense1 = Dense(g_dim, activation="relu")
        denseoutputs = []
        for j in range(0, len(object_combinations)):
            denseoutputs.append(dense1(outputs[j]))

        bn1 = BatchNormalization()
        bnoutputs = []
        for j in range(0, len(object_combinations)):
            bnoutputs.append(bn1(denseoutputs[j]))
        outputs = bnoutputs

    outputs = Add()(outputs)

    # f module
    for _ in range(f_layers):
        f = Dense(f_dim, activation="relu")(outputs)
        outputs = BatchNormalization()(f)

    f_out = Dense(out_dim)(outputs)

    return f_out
