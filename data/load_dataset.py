from abc import ABC, abstractmethod
import cv2


class DataLoader(ABC):

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_nth_batch(self, n):
        pass

    @abstractmethod
    def create_dataset_batch(self, n_batch_size):
        pass

    # TODO: tesztelni a referencia <-> érték sz átadás miatt
    def resize(self, dataset, annotation, size):
        height, width = size
        for i in range (0, len(dataset)):
            cv2.resize(src=dataset[i], dsize=(width, height), dst=dataset[i])
