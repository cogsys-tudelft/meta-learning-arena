# Meta-learning arena (`meta L arena`)

This repository contains a small framework for meta-learning experiments. It is based on PyTorch and PyTorch (Auto)Lightning.

## Installation

```bash
pip install git+ssh://git@github.com/V0XNIHILI/meta-learning-arena.git
```

## Usage

### Supervised learning

#### Speech commands V2

```bash
cd src/metalarena/configs/speech_commands
autolightning fit -c sc12_small.py -c your_local_settings.yaml
```

#### MNIST

```bash
cd src/metalarena/configs/mnist
autolightning fit -c seventy_k.py -c your_local_settings.yaml
```

### Metric learning

#### Omniglot

```python
cd src/metalarena/configs/omniglot
autolightning fit -c prototypical.py -c your_local_settings.yaml
```
