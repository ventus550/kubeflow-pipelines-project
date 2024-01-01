from kfp.dsl import component, Dataset, Input, Output, Model, Markdown
from src.secrets import configs

@component(base_image=configs.keras_image)
def visualize(model: Input[Model], plot: Output[Markdown]):                                                         
    
    print(f"Visualizing!")
    
    from src.utils import capture_image
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.plot(np.arange(10))
    img = capture_image()
    with open(plot.path, 'w') as f:
        f.write(f"![Image]({img})")