# A factorisation-aware matrix element emulator (FAME)

This is the project repository to accompany the article {}. In the article we describe in detail the construction of an emulator built using neural networks to emulate electron-positron annihilation into 5 jets.

Here we provide Python code to replicate our strategy.

## Requirements
FAME requires the following packages to run:
- Python 3.6+
- [NJet](https://bitbucket.org/njet/njet/wiki/Home) 2.1.0+
    - Other matrix element providers will also work, although we have interfaced with NJet in our examples.

## Installation
1. Clone the repo
2. Install with pip:
    ```
    pip install -e .
    ```
3. Point your to your NJet installation in:
    ```
    src/fame/utilities/njet_functions.py
    ```
    
## Usage
An example Jupyter notebook is provided in
```bash
src/fame/notebooks/quickstart_notebook.ipynb
```
which runs through the necessary steps to construct the emulator.

## License
Distributed under the [GPLv3](https://opensource.org/licenses/gpl-3.0.html) License.
