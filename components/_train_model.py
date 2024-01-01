from kfp.dsl import component, Dataset, Input, Output, Model
from src.secrets import configs

@component(base_image=configs.keras_image)
def train_model(epochs: int, dataset: Input[Dataset], oracle: Output[Model]):
    import numpy
    import keras
    import os
    
    from keras.layers import (
        Conv2D,
        BatchNormalization,
        MaxPooling2D,
        Dropout,
        Flatten,
        Dense,
        Activation,
        Permute,
        Reshape,
        Bidirectional,
        LSTM
    )
    from keras import callbacks
    import tensorflow as tf
    from src import aitoolkit

    image_width = 128
    image_height = 32
    
    # Character codes
    characters = [
        '!', '"', '#', '&', "'", '(', ')', '*', '+', ',',
        '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
        '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]

    X, Y = numpy.load(dataset.path + ".npz").values()
    maxlen = len(max(Y, key=len))
    X, Y = aitoolkit.format_data(X, Y, characters, maxlen)
    X_test, Y_test	 = X[:1000], Y[:1000]
    X_train, Y_train = X[1000:2000], Y[1000:2000] #todo
    
    
    tensorboard = callbacks.TensorBoard(
        # vertex-provided directory for logs
        log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR'],
        histogram_freq=1
    )
    checkpoints = callbacks.ModelCheckpoint(
        monitor='edit_distance',
        filepath='./weights.h5',
        save_weights_only=True,
        save_best_only=True,
        mode='min'
    )
    plateau = callbacks.ReduceLROnPlateau(
        monitor='edit_distance',
        patience=10,
        verbose=1,
        factor=0.90
    )

    namespace = tf.Graph()

    def Convolutions(filters, kernel=5):
        return keras.models.Sequential([
            Conv2D(filters, kernel, padding='same', kernel_initializer="he_normal"),
            BatchNormalization(),
            Activation('relu')
        ], name=namespace.unique_name("convolutions"))


    def build_model():
        input = keras.Input(shape=(image_height, image_width, 1), name="images")

        x = Convolutions(128, 3)(input)
        x = MaxPooling2D((2, 2))(x)

        x = Convolutions(64, 3)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Permute([2, 1, 3])(x)
        x = Reshape([image_width // 4, -1])(x)

        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)

        x = Bidirectional( LSTM(128, return_sequences=True, dropout=0.25) )(x)
        x = Bidirectional( LSTM(64,  return_sequences=True, dropout=0.25) )(x)

        # Reserve two extra tokens for the ctc_loss
        x = Dense(len(characters) + 2, activation="softmax", name="predictions")(x)

        model = keras.models.Model(inputs=input, outputs=x, name="oracle")
        model.compile(optimizer="adam", loss=aitoolkit.ctc_loss, metrics=[aitoolkit.edit_distance])
        return model
    
    model = build_model()
    print("Parameter count:", model.count_params())
    model.summary()


    model.fit(
        X_train, Y_train,
        validation_split = 0.1,
        epochs = epochs,
        callbacks = [tensorboard, checkpoints, plateau]
    )

    aitoolkit.save(model, f"{oracle.path}/model.h5", frozen = True)