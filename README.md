# eOHE-VAE

An algorithm to perform embedded One Hot Encoding (eOHE) on categorical data.

## Overview

This project uses the VAE model from SELFIES Project (https://github.com/aspuru-guzik-group/selfies/tree/master/original_code_from_paper/vae), and it has been modified to handle two different ways to perform eOHE, denominated eOHE-V1 and eOHE-V2.

## Info

Authors: Emilio Alexis de la Cruz Nuñez-Andrade, Isaac Vidal-Daza, James W. Ryan, Rafael Gómez-Bombarelli, Francisco J. Martín-Martínez


Swansea University, Massachusetts Institute of Technology, Universidad de Granada, 2023

## Installation

A `eohe_environment.yml` file has been provided with the necessary packages to run this project, to install it just run:

```bash
conda env create -f eohe_environment.yml
```
And you can activate it following the next command:

```bash
conda activate eohe
```

## eOHE Method

The description of Embedded One-Hot Encoding (eOHE) method can be found in our article:
``Embedded machine-readable molecular representation for more resource-efficient deep learning applications''



## Usage

To run the script, use the following command:

```bash
python rvae_rnn.py -encoding smiles -reduction0 -smiles_file smiles_file.csv
``` 


| Keyword      | Status    | Description                                                                         |
|--------------|-----------|-------------------------------------------------------------------------------------|
| -reduction0  | Mandatory | Choose one: -reduction0, -reduction1, -reduction2                                   |
| -reduction1  | Mandatory | Choose one: -reduction0, -reduction1, -reduction2                                   |
| -reduction2  | Mandatory | Choose one: -reduction0, -reduction1, -reduction2                                   |
| -encoding    | Mandatory | Choose one: smiles, deepsmiles, selfies                                             |
| -smiles_file | Mandatory | Path to the input file (e.g., smiles_file.csv)                                      |
| -nltanh      | Optional  | Change activation function in first layer of Encoder to Tanh instead of ReLU        |
| -scheduler   | Optional  | Add an scheduler with patience of 2 to reduce learning rate                         |

``smiles_file.csv`` file must not have header. 

For quick testing, please select any of the smiles files subsets from QM9_sizes, GDB_sizes or ZINC_sizes 
directories from QM9, GDB or ZINC databases, respectively.

For example
 ``-smiles_file QM9_sizes/qm9_smiles_size_1000_index_1.csv``

## Contributing

Please log bugs through GitHub.


