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

def createNNVar(train_inputs, train_context, valid_dataset, embeddings, nce_loss, optimizer, normalized_embeddings,
                similarity, init, valid_examples, doc_embeddings):
    nn_var = NNVar()
    nn_var.train_inputs = train_inputs
    nn_var.train_context = train_context
    nn_var.valid_dataset = valid_dataset
    nn_var.embeddings = embeddings
    nn_var.nce_loss = nce_loss
    nn_var.optimizer = optimizer
    nn_var.normalized_embeddings = normalized_embeddings
    nn_var.similarity = similarity
    nn_var.init = init
    nn_var.valid_examples = valid_examples
    nn_var.doc_embeddings = doc_embeddings
    return nn_var
