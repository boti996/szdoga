from tensorflow.keras.callbacks import BaseLogger
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data.load_training_data import DataLoader
from models.load_model import ModelLoader


def main():

    data_loader = DataLoader("./dataset", "./gts", 0.85)
    train_data, train_gt, test_data, test_gt = data_loader.create_dataset()

    size = (120, 120)
    data_loader.resize(train_data, size)
    data_loader.resize(train_gt, size)
    data_loader.resize(test_data, size)
    data_loader.resize(test_gt, size)

    model_loader = ModelLoader("my model")
    model = model_loader.get_model("./my model/pretrained-weights")

    batch_size = 32
    lr = 0.0001
    epochs = 1000

    model.compile(Adam(lr=lr), categorical_crossentropy)

    model.fit(train_data, train_gt, batch_size, epochs, verbose=1, validation_data=(test_data, test_gt), shuffle=True,
              callbacks=[BaseLogger])


main()
