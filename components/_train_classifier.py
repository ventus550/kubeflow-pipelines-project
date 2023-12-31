from kfp.dsl import component, Dataset, Input, Output, Model

@component(base_image="europe-central2-docker.pkg.dev/protocell-404013/kubeflow-images/keras:latest")
def train_classifier(epochs: int, dataset: Input[Dataset], classifier: Output[Model]):
    import numpy
    import keras
    import os
    
    from src.utils import save
    from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
    from keras import callbacks
    import tensorflow as tf

    X, Y = numpy.load(dataset.path + ".npz").values()
    X = X.reshape([-1, 70, 70, 1])
    test  = (X[:100], Y[:100])
    val = (X[100:1100], Y[100:1100])
    train = (X[1100:1200], Y[1100:1200]) #todo
    
    classes = ["other", "ellipse", "rectangle", "triangle"]
    
    # Preprocessing layers are bugged for some versions of keras
    # Using DataGenerator instead
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        shear_range=45.0,
        zoom_range = [1.2, 2],
        fill_mode='constant',
        cval=0,
    )

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", os.environ['AIP_TENSORBOARD_LOG_DIR'])
    tensorboard = callbacks.TensorBoard(
        # vertex-provided directory for logs
        log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR'],
        histogram_freq=1
    
    )
    checkpoints = callbacks.ModelCheckpoint(
        monitor='sparse_categorical_accuracy',
        filepath='./weights.h5',
        save_weights_only=True,
        save_best_only=True,
        mode='max'
    )
    plateau = callbacks.ReduceLROnPlateau(
        monitor='sparse_categorical_accuracy',
        patience=10,
        verbose=1,
        factor=0.90
    )
    
    model = keras.Sequential([
        keras.Input(shape=(70, 70, 1)),

        Conv2D(64, (5, 5), padding='same', activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(32, (5, 5), padding='same', activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Conv2D(32, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Flatten(),
        Dense(128, activation="relu"), 
        BatchNormalization(),
        Dense(len(classes), activation='softmax'),
    ])
    
    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=4e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
    
    history = model.fit(
        datagen.flow(*train, batch_size=64),
        validation_data=val,
        epochs=epochs,
        callbacks=[tensorboard, checkpoints, plateau]
    )

    save(model, f"{classifier.path}/model.h5", frozen = True)