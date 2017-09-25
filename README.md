# Gilt-TNR
This repository includes Python 3 implementations of the Gilt-TNR algorithm for different lattices.
The Gilt-TNR algorithm is described in the arXiv e-print "Renormalization of tensor networks using graph independent local truncations" available at https://arxiv.org/abs/1709.07460.
The implementations in this repository may remain under development, and no permanence is guaranteed.
For code that is guaranteed to reproduce the results in the aforementioned paper, see the ancillary files of the arXiv submission (directory `anc` in the tarball https://arxiv.org/src/1709.07460).

The square lattice version of Gilt-TNR is implemented in GiltTNR2D.py.
This implementation is fully functional and produces accurate physical observables for the models we have tested it on.
Further development may or may not happen.

Work for a cubical lattice version is ongoing in GiltTNR3D.py
The code is in a runnable state, and it tries to perform a Gilt-TNR step on a given model.
However, bugs are still possible, even probable, performance needs significant improvement, and the design of the algorithm may still change.

All the source code is licensed under the MIT license, as described in the file LICENSE.

For any questions or comments, please email Markus Hauru at markus@mhauru.org.

## Installation
First of all, you need Python 3 and a recent version of scipy and numpy.
(The code has been tested with Python 3.5.3, scipy 0.18.1 and numpy 1.12.1.)
An easy way to get these running is to use a scientific Python distribution, such as anaconda: https://www.anaconda.com/download/

Once you have Python running, you need to fetch this repository.
On a *nix system you can simply run<br>
```git clone --recursive  https://github.com/Gilt-TNR/Gilt-TNR```<br>
If you run `git clone` without the `--recursive`, then you'll probably be missing the git submodules that include libraries that the Gilt-TNR algorithm depends on.
You can get them afterwards by running `git submodule update --init --recursive` in your repository.

Similar commands should be available on non-nix systems, if git is available.

If you do not have git, you can simply download the repository as zip or a tarball.
You will, however, manually have to download the three libraries that this repository depends on, and place them in the same folder as the main code files, or somewhere where your `PYTHONPATH` environment variable can find them.
The three repositories you need are:<br>
https://github.com/mhauru/tensors<br>
https://github.com/mhauru/ncon<br>
https://github.com/mhauru/tntools

## A quick guide to the code

The main files that implement the square and cubical versions of the algorithm are `GiltTNR2D.py` and `GiltTNR3D.py`.
For documentation on how the code works, see the source code itself.

For each of the algorithms the following also exist.

A `_envspec.py` file<br>
A script that runs the Gilt-TNR algorithm to produce coarse-grained tensors, and then gets the environment spectrum for a given environment, and prints it out.

A `_test.py` file<br>
A script that runs the Gilt-TNR algorithm on a given lattice model, producing coarse-grained tensors, and evaluating and printing out various things, such as free energies and spectra of the coarse-grained tensors.

A `_setup.py` file<br>
A module for interfacing for with `tntools.datadispenser` module.
This is usually of no concern to the user.
See the source code for `tntools.datadispenser` for more details.

There's also the file `GiltTNR3D_impurity.py` which applies the GiltTNR3D algorithm to an impurity.
This remains unfinished at the moment.

To run the scripts, commands of the following form can be used:<br>
```python3 GiltTNR2D_test.py -c 'confs/GiltTNR2D_test_batch.yaml' -y 'gilt_eps: !!float 1e-7' 'beta: 0.4'```<br>
The command line argument `-c` specifies a configuration file, that lists values for the various parameters that the `_test.py` script and the algorithm itself take.
These files use the YAML format.
Parameters in a configuration file can be appended or overriden by providing more command line arguments after the `-y` flag.
Each argument should be a string, that could be added as a line to the YAML file.
The various parameters that can be used are defined in the beginning of the scripts and in the docstrings in the main algorithm files.

The code makes significant use of lower level tools from the three packages that it has as submodules: `tensors`, `ncon` and `tntools`. `tensors` is a library for implementing basic tensor operations, such as decompositions and contractions. The key benefit over numpy's `ndarray`s (which it uses under the hood) is that `tensors` supports tensors with internal Abelian symmetries (see https://arxiv.org/abs/1008.4774). `ncon` is a Python implementation of the NCon function, as described here: https://arxiv.org/abs/1402.0939. `tntools` is a collection of miscellaneous tools useful when working with tensor network algorithms. See the repositories themselves for more documentation.

Note that all the actual running of the coarse-graining algorithms, and generating coarse-grained tensors, happens using a module called `tntools.datadispenser`. The user interface is mainly through the function `datadispenser.get_data`, for which usage examples can be seen in the `_test.py` and `_envspec.py` scripts. The idea is that the user specifies the type of data (typically just `A`, which is used as the name for coarse-grained tensors as in the paper, or `As` for the algorithms were several A tensors are needed, such as GiltTNR3D) and the parameters for creating this data (in a dictionary). `datadispenser` then generates this data, and stores it on disk, so that the next time the same data is requested, `datadispenser` just finds it on the disk and returns it from there. Note that if one edits the code, either the old data should be purged, or the version number of the algorithm should be changed, so that `datadispenser` knows the regenerate all data that is requested. See the docstring for `datadispenser` and its functions for more details.
