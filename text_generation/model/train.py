
import tensorflow as tf
from text_generation.model.text_gen_model import  TextGenerateModel
import  text_generation.model.test_gen_config
from text_generation.model.test_gen_config import TextGenConfig
from text_generation.data.text_gen_data_config import TextGenDataConfig
from text_generation.data.text_gen_data_loader import TextGenDataLoader

def _train(model,data_loader,config):

   model.run_train()

def main(unused_argv):
    data_config = TextGenDataConfig(model_name="TextGeneration")
    data_loader = TextGenDataLoader(data_config=data_config)
    config = TextGenConfig(model_name="TextGeneration")
    model=TextGenerateModel(config=config,data_loader=data_loader)


    _train(model,data_loader,config)

if __name__ == '__main__':
    tf.app.run()