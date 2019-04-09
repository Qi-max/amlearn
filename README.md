# <img alt="amlearn" src="docs_rst/_static/amlearn_logo.png" width="300">
Machine Learning Package for Amorphous Materials.

To featurize the heterogeneous atom site environments in amorphous materials, we establish a comprehensive representation (comprising of 740+ site features) that encompass short- (SRO) and medium-range order (MRO). We integrate Fortran90 with Python (using f2py) to achieve combination of the
flexibility and fast-computation (>10x times faster than pure Python) of features. Please see examples
in the SRO and MRO representations in the `amlearn.featurize`. 

Please see more details in a recent paper from us: [Qi Wang and Anubhav Jain. Linking plastic heterogeneity of bulk metallic glasses to quench-in structural defects with machine learning](https://arxiv.org/abs/1904.03780). [arXiv:1904.03780](https://arxiv.org/abs/1904.03780)


<div align='center'><img alt="amlearn" src="docs_rst/_static/schematic_ML_of_deformation.png" width="800"></div>   
&nbsp;

In addition, wrapper classes and utility functions for featurizers powered by matminer/amp and machine
learning algorithms supported by scikit-learn are also included.         


## Installation

Before installing amlearn, please install numpy (version 1.7.0 or greater) first.

We recommend to use the conda install.

```sh
conda install numpy
```

or you can find numpy installation guide from [Numpy installation instructions](https://www.scipy.org/install.html).


Then, you can install amlearn. There are two ways to install amlearn:

**Install amlearn from PyPI (recommended):**

```sh
pip install amlearn
```


**Alternatively: install amlearn from the GitHub source:**

First, clone amlearn using `git`:

```sh
git clone https://github.com/Qi-max/amlearn
```

 Then, `cd` to the amlearn folder and run the `setup.py`:
```sh
cd amlearn
sudo python setup.py install
```

