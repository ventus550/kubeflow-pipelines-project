from kfp.dsl import component, Dataset, Input, Output, Model, Markdown
from src.secrets import configs

@component(base_image=configs.keras_image, packages_to_install=["seaborn"])
def metrics(
    dataset: Input[Dataset],
    model: Input[Model],
    edit_distance_histogram: Output[Markdown],
    predictions: Output[Markdown]
) -> int:                                                         
    
    from src import aitoolkit
    from src.utils import capture_image
    import seaborn
    import numpy
    
    X, Y = numpy.load(dataset.path + ".npz").values()
    X, Y = aitoolkit.format_data(X, Y, aitoolkit.characters)
    model = aitoolkit.load(model.path)
    batch = aitoolkit.batch_prediction(model, X, Y)
    
    edit_distance_values = np.array([sample.value for sample in batch])
    seaborn.histplot(edit_distance_values, bins=10, kde=True, alpha=0.6)
    open(edit_distance_histogram.path, 'w').write(f"![Image]({capture_image()})")
    
    batch.show()
    batch.show()
    open(predictions.path, 'w').write(f"![Image]({capture_image()})")
    
    average_edit_distance = np.mean(edit_distance_values)
    print("Average edit distance:", average_edit_distance)
    return average_edit_distance