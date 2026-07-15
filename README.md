# multilayer-perceptron

An introduction to artificial neural networks through the implementation of a
multilayer perceptron.

## Launching

Launch the training
```bash
python train.py network.txt train.csv validation.csv
```

Example output
```
...
[mlp/fit] epoch 800/800 - accuracy: 0.989 loss: 0.0508 - val_accuracy: 0.974 val_loss: 0.0998

[train] validation accuracy:         97.368%
[train] binary cross-entropy (eval): 0.0998

[train] model saved   - network.pkl
[train] history saved - network.json
[train] plot saved    - network.png
```

Evaluate the model
```bash
python evaluate.py network.pkl validation.csv
```
Example output
```
[evaluate] .\validation.csv - 114 entries  shape (114, 30)
[evaluate] .\network.pkl - 4 layers  800 epochs  455 batch_size

[evaluate] accuracy:              97.368%
[evaluate] precision:             97.826%
[evaluate] binary cross-entropy:  0.0998

[evaluate] of 47 cancer cases:  45 detected  (95.7%), 2 missed (4.3%)
[evaluate] of 67 benign cases:  66 detected  (98.5%), 1 false positive (1.5%)
```

## Setup

[uv](https://github.com/astral-sh/uv) is used to install and manage python,
it is simple to install on school computers and creates the venv just fine.

```bash
uv sync # makes the venv and installed packages

# Linux
source ./venv/bin/active # set the python to current venv

# Windows
.\.venv\Scripts\activate
```

[Jupyter notbook](https://jupyter.org/) is a nice mix of docs & code, to launch it:

```zsh
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
