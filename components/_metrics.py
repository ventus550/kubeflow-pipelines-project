from kfp.dsl import component, Dataset, Input, Output, Model, Markdown
from src.secrets import configs

@component(base_image=configs.keras_image)
def upload_model(model: Input[Model]):                                                         
    
    from src import aitoolkit
    model = aitoolkit.load(f"{model.path}/model.h5")
    print("Upload finished")