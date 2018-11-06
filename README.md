Solution to the first project of the deep reinforcement learning nanodegree at Udacity.

## Problem definition

In this environment, a double-jointed arm can move to target locations through a 3d space. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Usage

### Preparation

Please ensure you have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed. Clone the repository and use `pipenv --three install` to create yourself an environment to run the code in. Otherwise just install the packages mentioned in Pipfile.

Due to the transitive dependency to tensorflow that comes from unity ml-agents and the [bug](https://github.com/pypa/pipenv/issues/1716) causing incompatibility to jupyter you might want to either drop the jupyter from the list of dependencies or run `pipenv --three install --skip-lock` to overcome it.

To activate a virtual environment with pipenv issue `pipenv shell` while in the root directory of the repository.

After creating and entering the virtual environment you need to set a `DRLUD_P2_ENV` shell environment which must point to the binaries of the Unity environment. Example of for Mac OS version of binaries it might be 
```
DRLUD_P2_ENV=../deep-reinforcement-learning/p2_navigation/Reacher.app; export DRLUD_P2_ENV
```

Details of downloading and setting of the environment are described in Udacity nanodegree materials.

### Training (easy way)

Just follow the [training notebook](Training.ipynb).

### Training (proper way)

The executable part of code is built as a three-stage pipeline comprising of
* training pipeline
* analysis notebook
* demo notebook

The training pipeline was created with the idea of helping the researcher to keep track of his experimentation process as well as keeping the running results. The training process is spawned by executing the `trainer.py` script and is expected to be idempotent to the training results, i.e. if the result for a specific set of hyperparameters already exists and persisted, the trainer will skip launching a training session for this set of hyperparameters.

The sets of hyperparameters for training are defined inside of `trainer.py` in the `simulation_hyperparameter_reference` dictionary which is supposed to be append-only in order to keep consistency of the training result data. Each of the hyperparameters sets will produce a file with a scores of number of runs of an agent which will be stored inside of `./hp_search_results` directory with an id referring to the key from the `simulation_hyperparameter_reference` dictionary. The neural networks weights for every agent training run will be stored in the same directory with the relevant hyperparameters key as well as random seed used.

To train an agent with a new set of hyperparameters just add an item into `simulation_hyperparameter_reference` object. Here's an example of adding an item with id `25` after existing hyperparameter set with id `24`:

```
simulation_hyperparameter_reference = {
###
###  Skipped some items here
###

    24: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 1e-5, 0,    False, False, 1)),
    25: launch_parm("ddpg",  1, ddpg_parm(int(1e5), 256,  0.99, 1e-3, 1e-4, 2e-5, 0,    False, False, 1)),
}
```

The set of hyperparameters is represented as an instance of a namedtuple `launch_parm` which has the following set of fields:

* algorithm
* times
* algorithm-specifgic set of hyperparameters

The `algorithm` field defines an implementation of an agent, currently only `ddpg` algorithm is supported.

DDPG-related hyperparameters are passed using `ddpg_parm` namedtuple and contain the following fields:

* memory_size - number of experiences to persist
* batch_size - numer of experiences to use during a learning session
* gamma
* tau
* lr_actor
* lr_critic
* weight_decay
* noise_enabled
* gradient_clipping
* learn_every - number of steps between learning sessions

The meaning and effects of other values for these field are discussed in the [hyperparameter search notebook](Training_hyperparameter_search_analysis.ipynb). 

## Implementation details

Two neural network architectures are defined in the `models.py` file. 
* Actor class implement a three-layer neural network that learns to project the state on to an action.
* Critic class implements a three-layer neural network that learns to map state and an action into a Q-value.

## Results 

Please check the [following notebook](Report.ipynb) for the best set of hyperparameters I managed to identify.
