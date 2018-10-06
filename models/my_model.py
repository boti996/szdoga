from models.model import Model
from keras.models import Sequential


class MyModel(Model):

    def get_model(self):
        model = Sequential()
        # TODO: structure
        return model
