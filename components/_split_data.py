from kfp.dsl import component, Dataset, Input, Output

@component(base_image="europe-central2-docker.pkg.dev/protocell-404013/kubeflow-images/keras:latest")
def split_data(ratio: float, dataset: Input[Dataset], train: Output[Dataset], test: Output[Dataset]):
    import numpy
    X, Y = numpy.load(dataset.path).values()
    test_size = int(ratio * len(X))
    
    train_set = (X[test_size:], Y[test_size:])
    test_set  = (X[:test_size], Y[:test_size])
    
    numpy.savez(train.path, *train_set)
    numpy.savez(test.path, *test_set)