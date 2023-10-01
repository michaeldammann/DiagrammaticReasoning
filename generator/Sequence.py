class Sequence(object):

    def __init__(self, seq_type, seq_inner_type):
        self.seq_type = seq_type
        self.seq_inner_type = seq_inner_type
        self.symbols = []
        self.inner_symbols = []

    def __repr__(self):
        return str(self.symbols)+str(self.inner_symbols)

