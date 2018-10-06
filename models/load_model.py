from models.my_model import MyModel


class ModelLoader(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.types = self.init_model_types()

        self.model = self.types.get(model_name)

    def init_model_types(self):
        types = {
            "my model": MyModel()
        }

        return types

    def get_model(self, weights=None):
        model = self.model.get_model()

        if weights:
            model.load_weights(filepath=weights, by_name=True)

        return model

