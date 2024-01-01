from kfp.dsl import component, Dataset, Input, Output
from src.secrets import configs

@component(base_image = configs.keras_image)
def split_data(ratio: float, dataset: Input[Dataset], train: Output[Dataset], test: Output[Dataset]):
    import numpy
    X, Y = numpy.load(dataset.path).values()
    test_size = int(ratio * len(X))
    
    train_set = (X[test_size:], Y[test_size:])
    test_set  = (X[:test_size], Y[:test_size])
    
    numpy.savez(train.path, *train_set)
    numpy.savez(test.path, *test_set)