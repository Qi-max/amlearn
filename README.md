# <img alt="amlearn" src="docs_rst/_static/amlearn_logo.png" width="300">
Machine Learning Package for Amorphous Materials.

To featurize the heterogeneous atom site environments in amorphous materials,
we can use `amlearn` to derive 1k+ candidate features that encompass short- (SRO)
and medium-range order (MRO) to describe the packing heterogeneity around each atom site. 
(See the following example figure for combining site features and machine learning (ML) to predict the 
deformation heterogeneity in metallic glasses). 

Candidate features include recognized signatures
such as coordination number (CN), Voronoi indices, characteristic motifs,
volume metrics (atomic/cluster packing efficiency), i-fold symmetry indices,
bond-orientational orders, symmetry functions (originally proposed to fit
ML interatomic potentials and recently gained success in featurizing disordered
materials), as well as our recently proposed highly interpretable and generalizable
distance/area/volume interstice distribution features (to be published).

In `amlearn`, We integrate Fortran90
with Python (using f2py) to achieve combination of the flexibility and
fast-computation (>10x times faster than pure Python) of features.
Please refer to the SRO and MRO feature representations in `amlearn.featurize`.


<div align='center'><img alt="amlearn" src="docs_rst/_static/schematic_ML_of_deformation.png" width="800"></div>   
&nbsp;

In addition, wrapper classes and utility functions for machine
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

