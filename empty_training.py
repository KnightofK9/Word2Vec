class EmptyTraining:
    def __init__(self):
        self.train_data = None
        self.train_data_saver = None

    def train(self):
        config = self.train_data.config
        word_mapper = self.train_data.word_mapper
        for (batch_inputs, batch_context) in self.train_data:
            for batch_count in range(0, config.batch_size):
                word = batch_inputs[batch_count]
                context = batch_context[batch_count]
                id = ""
                if config.is_doc2vec():
                    id = "|{}".format(word[-1])
                word_list = list(map(word_mapper.id_to_word,word[:-1]))
                word_context = word_mapper.id_to_word(context[0])
                print("{}{} -> {}".format(word_list,id,word_context))

    def set_train_data(self, train_data, train_data_saver):
        self.train_data = train_data
        self.train_data_saver = train_data_saver

    def restore_last_training_if_exists(self):
        pass
