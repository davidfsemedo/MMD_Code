# Multimodel Dataset

This project is divided into two main parts: The __Image__ part and the __Text__ part. 
Although similar they implement different models as described in Saha et. al.

## Structure of the project

The main folder must contain an _image_annoy_index_ folder, a _multimodal_hred_image_task_ 
and _multimodal_hred_text_task_ folder. The main folder must also contain, and in the same level, a _Target_model_ function.

The __Target_model__ folder is where the system will store all it's output files 
(namely the vocabulary file, the training/validation/test data files, and the tensorflow files containing the best trained model so far).


## Running the Models

Contrary to what is said in the README files inside multimodal_hred_image_task/multimodal_hred_text_task folder, to run the model you need to edit
the `run_model_task<1/2>.py` and set the `data_dir` variable to the path to where the dataset is located (specifying one of the available versions (v1 or v2)) and set the `dump_dir` to the location of the of the __Target_model__ folder.

After correctly specifying the paths simply run `python run_model_task<1/2>.py` and the model should start training and storing the best models in Target_model/model

## Running the chatbot prediction server

In multimodal_hred_text_task/run_predictions.py is implemented a simple server that listens to a user query, encodes it and sends it
to the model to obtain a response from it and return it to the user. This server can be linked to a chatbot UI in order to implement a replica of the chat application depicted in Saha et. al.

This file provides a useful guide as to how to encode a single user utterance and obtain a prediction for it.



## Environment setup
```conda create --name mmd_code python=3.7
conda install nltk
conda install tensorflow-gpu=1.15
pip install annoy
pip install orjson
```
