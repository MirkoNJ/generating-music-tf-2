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
│   │
│   └─── < composer_name >                  <- Contains all .mid files used
|                                              for composer with composer_name
│   
└───data_info
│   chopin_title_opus.csv                   <- A mapping from Opus to Title 
|                                              of pieces by F. Chopin
│    
└───notebooks
|   │   modelling.ipynb                     <- Preprocessing and Training
|   │   model_results_visualisation.ipynb   <- Results Analysis
|   │   music_generation.ipynb              <- Generating new music
|   │   playground.ipynb                    <- Trying out different stuff
│   │
│   └───modules                     
|       |   batch.py                        <- generating batches
|       |                                       for training
|       |   midi_related.py                 <- everything necessary
|       |                                       for .mid import/export
|       |   plotting.py                     <- everything necessary 
|       |                                       for plotting results
|       |   preprocessing.py                <- transforms note state matrices
|       |                                       to inputs
|       |   subclasses.py                   <- custom layers, metrics
|       |                                       and loss function
│   
└───outputs
│   │
│   └───midi 
|   |   |   
|   |   └───train                    <- .mid files generated during training
|   |   |   |    < model_name >.mid            
|   |   |   
|   |   └───generated                <- .mid files generated after training
|   |   |   |    < model_name >.mid            
|   |   
|   |   └───results                  <- Renamed and collected 
|   |   |   |                           .mid files based on data used for model
|   |   |   |    < model_name >.mid            
|   |   
│   └───models
|   |   |   
|   |   └───arrays                   <- numpy arrays saved after training
|   |   |   |    < model_name >.mid            
|   |   |   
|   |   └───ckpt                     <- saved models during and after training
|   |   |   |    < model_name >.mid            

```

## Using it

Run the ```notebook/modelling.ipynb ```to train a model and save a model.
Load the saved model in ```notebook/music_generation.ipynb ``` to generate new .mid files.