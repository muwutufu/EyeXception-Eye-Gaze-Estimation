# EyeXception: Gaze Estimation with Deep Learning

This project is collaborated with Peter Grönquist (petergro@student.ethz.ch). Both authors contributed equally to this work.

We explore different classification model adaptations
stemming from deep learning, whilst combining it with methods
from computer vision. We show that by using the combined methods and 
applying a statistical evaluation it is possible to obtain state
of the art predictions in eye-gaze approximation.

## Setup
To obtain the results from our submission follow these steps:

- Install the dependencies, we only use tqdm as an additional dependency to the ones provided in setup.py
- Clone the repository
- Run preprocessing.py
- Adapt the paths in parameters.py to match the dataset path
- Copy the repository at least 10 times
- Run train.py in each
- After training is finished, run generate.py
- Gather all predictions from the respective files in ```./tf/```
- Take the mean of all predictions

### Necessary datasets

Either run preprocessing.py on the provided h5 datasets or ask for them, we will happily make them available on a harddisk or by other means on leonhard, as polybox is limited to 50GB.

### Installing dependencies

Run (with `sudo` appended if necessary),
```
python3 setup.py install
```
and additionally install tqdm

Note that this can be done within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). In this case, the sequence of commands would be similar to:
```
    mkvirtualenv -p $(which python3) myenv
    python3 setup.py install
```

when using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

## Structure

* `src/` - all source code.
    * `Dated_Training/` - dated training files for old neural networks
    * `models/` - neural network definitions
    * `tb/` - Tensorboard logs
    * `tf/` - Tensorflow trained models and predictions
    * `util/` - utility methods
    * `generate.py` - generator for test dataset script
    * `Mean.py` - simple script to get the mean of 10 predictions
    * `parameters.py` - parameters file
    * `preprocessing.py` - preprocessing script
    * `train.py` - training script


### Outputs
Once training is complete for all models and the predictions have been generated, you will find each individual prediction in the corresponding ```./tf/``` folder
