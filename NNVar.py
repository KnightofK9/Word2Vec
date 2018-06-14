class NNVar:
    def __init__(self):
        self.train_inputs = None
        self.train_context = None
        self.valid_dataset = None
        self.embeddings = None
        self.nce_loss = None
        self.optimizer = None
        self.normalized_embeddings = None
        self.similarity = None
        self.init = None
        self.valid_examples = None
        self.doc_embeddings = None

class CNN_Var:
    def __init__(self):
        self.train_inputs = None
        self.train_context = None
        self.dropout_keep_prob = None
        self.valid_dataset = None
        self.loss = None
        self.train_op = None
        self.accuracy = None
        self.init = None
        self.h_pool_flat = None
        self.correct_predictions = None

