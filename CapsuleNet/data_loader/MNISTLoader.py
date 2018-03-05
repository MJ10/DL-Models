from tensorflow.examples.tutorials.mnist import input_data


class MNISTLoader:
    def __init__(self, config):
        self.config = config
        self.data = input_data.read_data_sets('../../MNIST_data/')

    def next_batch(self, batch_size):
        return self.data.train.next_batch(batch_size)

    def next_batch_val(self, batch_size):
        return self.data.validation.next_batch(batch_size)
