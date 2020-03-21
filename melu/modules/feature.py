class SparseFeature(object):
    def __init__(self, vocab_size, dim, name, one_hot=True):
        self.vocab_size = vocab_size
        self.dim = dim
        self.name = name
        self.one_hot = one_hot

