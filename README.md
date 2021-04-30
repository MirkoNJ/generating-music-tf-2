# Generating Music

An implementation of a bi-axial LSTM (Tensorflow 2) very similar to [Nikhil Kotecha's adaption](https://github.com/nikhil-kotecha/Generating_Music) (Tensorflow 1) of [Daniel D Johnson's bi-axial LSTM](https://github.com/danieldjohnson/biaxial-rnn-music-composition) (Theano) as described in  [this blog post](https://www.danieldjohnson.com/2015/08/03/composing-music-with-recurrent-neural-networks/) .

## Requirements

First ensure that Python (version 3.8) is installed.

### Set up using pipenv

```
cd /path/to/this/project
pipenv install 
```


### Set up using pip

Install the dependencies listed in the Pipfile:

```
pip install ipywidgets notebook pandas plotly pydot tensorflow-addons tensorflow-gpu tensorflow-probability  
pip install git+https://github.com/louisabraham/python3-midi#egg=midi
```

## Directory structure 

```
generating-music-tf-2
│   README.md
│   LICENSE.md
│   Pipfile
|   Pipfile.lock   
│
└───data
│   │   chopin_title_opus.csv           <- For mapping .mid files to opus number
│   │   sources_for_chopin_midi.txt     <- Sources for .mid files
│   │
│   └───chopin_midi                     <- All chopin .mid files
│   
└───notebooks
|   │   main.ipynb                      <- Data preperation, training, music generation
|   │   playground.ipynb                <- Trying out different stuff
│   │
│   └───modules                     
|       |   batch.py                    <- generating batches for training
|       |   midi_related.py             <- everything necessary for .mid import/export
|       |   preprocessing.py            <- transforming note state matrices to inputs
|       |   subclasses.py               <- custom layers, metrics and loss function
│   
└───outputs
│   │
│   └───midi 
|   |   |   
|   |   └───train                       <- .mid files generated during training
|   |   |   |   model_name_1            
|   |   |   |   model_name_2             
|   |   |   |   ...
|   |   |   
|   |   └───generated                   <- .mid files generated after training
|   |   |   |   model_name_1            
|   |   |   |   model_name_2             
|   |   |   |   ...
|   |   
│   └───models
|   |   |   
|   |   └───arrays                      <- numpy arrays saved during training
|   |   |   |   model_name_1            
|   |   |   |   model_name_2             
|   |   |   |   ...
|   |   |   
|   |   └───ckpt                        <- saved models during and after training
|   |   |   |   model_name_1            
|   |   |   |   model_name_2            
|   |   |   |   ...

```

## Using it

Run the ```notebook/main.ipynb ```to train a model and generate new .mid files.
(A more detailed description of how to customize, will hopefully follow here.)
