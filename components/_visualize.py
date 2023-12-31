from kfp.dsl import component, Dataset, Input, Output, Model, Markdown

@component(base_image="europe-central2-docker.pkg.dev/protocell-404013/kubeflow-images/keras:latest")
def visualize(model: Input[Model], plot: Output[Markdown]):                                                         
    
    print(f"Visualizing!")
    
    from src.utils import capture_image
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.plot(np.arange(10))
    img = capture_image()
    with open(plot.path, 'w') as f:
        f.write(f"![Image]({img})")