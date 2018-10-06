class DataLoader(object):

    def __init__(self, dataset, ground_truth, rate_of_training_set):
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.rate = rate_of_training_set

    def create_dataset(self):
        # returns: train_data, train_gt, test_data, test_gt

    def create_dataset_batch(self, batch_size, n_batch):
        # same with batches

    def resize(self, dataset, width, height):
        self.resize(dataset, (width, height))

    def resize(self, dataset, size):
        # resize image array

    def reduce_channels(self, type, max_val=255):
        # type: r, g, b, rg, rb, gb
        # max_val: bits per channel (31, 63, 127, 255)


