# multilayer-perceptron

An introduction to artificial neural networks through the implementation of a
multilayer perceptron.


## Setup

[uv](https://github.com/astral-sh/uv) is used to install and manage python,
it is simple to install on school computers and creates the venv just fine.

### Launch Jupyter

[Jupyter notbook](https://jupyter.org/) is a nice mix of docs & code, to launch it:

```zsh
uv sync # makes the venv and installed packages
source ./venv/bin/active
jupyter notebook
```

## Notions for the evaluation

Must understand the training phase (learning pase) and the underlying
algorythms.

- feedforward
- backprogagation
- gradient decent

## Dataset

It's a csv file with 32 columns extracted from a real dataset of cell
characteristics based on fine-needle aspiration, more info
[WDBC](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names).

The goal is to predic a cancer diagnosis, the M,B colum in the dataset
(malignant or benign). The data needs proprocessing before being used.

## Implementation


- A split datasets program
- A train program
- A predict program
- The training process must be configurable, via commandline or input file.
- It must track prgress for a loss & accuracy graph at the end.
- Each epoch must print loss, and val\_loss
- Weights must be outputted to a file.
- Use a seed value for training.


## Nural network
Concept seems simple, but implementation not so much, the google doc is good
place to start.

## Links

- [google ml crashcourse](https://developers.google.com/machine-learning/crash-course/neural-networks)
- [classification and manifolds](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

