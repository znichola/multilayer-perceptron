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

### PyTorch interface

I will copy the pytorch interface, seems like a reasonable appraoch, and their docs hace nice clickable links in code examples. [build pyTorch model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

The overall layout is something like this:

```python
class Module:
    # something
    self.sequential_stack # the stack of Layers to calculate
    def forward(vecInput) -> vecOutput: #doing the layer calcs
        pass

class Layer:
    # Applies transform to incomming data : y = x * A^T + b
    def forward(vecInput) -> vecOutput:
        pass

class ReLU:
    # Applies the rectified linear unit function element-wise : ReLU(x) = max(0, x)
    def forward(vecInput) -> vecOutput:
        pass

```

## Links

- [google ml crashcourse](https://developers.google.com/machine-learning/crash-course/neural-networks)
- [classification and manifolds](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
- [categorical cross entropy - kera git](https://github.com/keras-team/keras/blob/465f93f4cc6511da3bb327e49583099eafe0753c/keras/src/backend/tensorflow/nn.py#L1110)
