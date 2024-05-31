# dpkernel

This package implements the differentially private dpMMD and dpHSIC tests for two-sample and independence testing, as proposed in our paper [Differentially Private Permutation Tests: Applications to Kernel Methods](https://arxiv.org/abs/2310.19043).

The implementation is in [JAX](https://jax.readthedocs.io/) which can leverage the architecture of GPUs to provide considerable computational speedups.

The experiments of the paper can be reproduced using the [dpkernel-paper](https://github.com/antoninschrab/dpkernel-paper/) repository, which also contains a [demo.ipynb](https://github.com/antoninschrab/dpkernel-paper/blob/master/demo.ipynb) notebook explaining how to use dpMMD and dpHSIC.

## Installation

The `dpkernel` package can be installed by running:
```bash
pip install git+https://github.com/antoninschrab/dpkernel.git
```
which relies on the `jax` and `jaxlib` dependencies.

In order to run the tests on GPUs the `cuda` versions of [JAX](https://jax.readthedocs.io/en/latest/installation.html) should be installed as follows
```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
This can also be run before installing `dpkernel`.

## Examples

**Jax compilation:** The first time the dpmmd or dphsic functions are evaluated, JAX compiles them. 
After compilation, they can fastly be evaluated at any other X and Y of the same shape, and any epsilon. 
If the functions are given arrays with new shapes, the functions are compiled again.
For details, check out the [demo.ipynb](https://github.com/antoninschrab/dpkernel-paper/blob/master/demo.ipynb) notebook on the [dpkernel-paper](https://github.com/antoninschrab/dpkernel-paper/) repository.

### dpMMD

**Two-sample testing:** Given arrays X of shape $(m, d)$ and Y of shape $(n, d)$, our dpMMD test `dpMMD(key, X, Y, epsilon, delta)` returns 0 if the samples X and Y are believed to come from the same distribution, or 1 otherwise, and is (epsilon, delta) differential private.

```python
# import modules
>>> import jax.numpy as jnp
>>> from jax import random
>>> from dpkernel import dpmmd, dphsic, human_readable_dict

# generate data for two-sample test
>>> key = random.PRNGKey(0)
>>> subkeys = random.split(subkey, num=2)
>>> X = random.uniform(subkeys[0], shape=(500, 10))
>>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1

# run dpMMD test
>>> key, subkey = random.split(key)
>>> output = dpmmd(subkey, X, Y, epsilon=0.7, delta=0.1)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = dpmmd(subkey, X, Y, epsilon=0.7, delta=0.1, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'Bandwidth': 3.1622776601683795,
 'DP delta': 0.1,
 'DP epsilon': 0.7,
 'Kernel gaussian': True,
 'Non-privatised MMD V-statistic': 1.0208993368311252,
 'Number of permutations': 2000,
 'Privacy Laplace noise for MMD V-statistic': 0.003230674754896666,
 'Privatised MMD V-statistic': 1.024130011586022,
 'Privatised MMD quantile': 0.07137244880436107,
 'Privatised p-value': 0.0004997501382604241,
 'Privatised p-value threshold': 0.05,
 'Test level': 0.05,
 'dpMMD test reject': True}
```

### dpHSIC

**Independence testing:** Given paired arrays X of shape $(n, d_X)$ and Y of shape $(n, d_Y)$, our dpHSIC test `dpHSIC(key, X, Y, epsilon, delta)` returns 0 if the paired samples X and Y are believed to be independent, or 1 otherwise, and is (epsilon, delta) differential private.

```python
# import modules
>>> import jax.numpy as jnp
>>> from jax import random
>>> from dpkernel import dpmmd, dphsic, human_readable_dict

# generate data for independence test 
>>> key = random.PRNGKey(0)
>>> subkeys = random.split(subkey, num=2)
>>> X = random.uniform(subkeys[0], shape=(500, 10))
>>> Y = X + 0.01 * random.uniform(subkeys[1], shape=(500, 10))

# run dpHSIC test
>>> key, subkey = random.split(key)
>>> output = dphsic(subkey, X, Y, epsilon=0.7, delta=0.1)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = dphsic(subkey, X, Y, epsilon=0.7, delta=0.1, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'Bandwidth X': 3.1622776601683795,
 'Bandwidth Y': 3.1622776601683795,
 'DP delta': 0.1,
 'DP epsilon': 0.7,
 'Kernel X gaussian': True,
 'Kernel Y gaussian': True,
 'Non-privatised HSIC V-statistic': 0.04508960829602034,
 'Number of permutations': 2000,
 'Privacy Laplace noise for HSIC V-statistic': 0.009119452651766512,
 'Privatised HSIC V-statistic': 0.05420906094778685,
 'Privatised HSIC quantile': 0.05035574742299952,
 'Privatised p-value': 0.04097951203584671,
 'Privatised p-value threshold': 0.05,
 'Test level': 0.05,
 'dpHSIC test reject': True}
```

## Contact

If you have any issues running our dpMMD and dpHSIC tests, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@unpublished{kim2023differentially,
title={Differentially Private Permutation Tests: {A}pplications to Kernel Methods}, 
author={Ilmun Kim and Antonin Schrab},
year={2023},
url = {https://arxiv.org/abs/2310.19043},
eprint={2310.19043},
archivePrefix={arXiv},
primaryClass={math.ST}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).

## Related tests

- [mmdagg](https://github.com/antoninschrab/mmdagg/): MMD Aggregated MMDAgg test 
- [ksdagg](https://github.com/antoninschrab/ksdagg/): KSD Aggregated KSDAgg test
- [agginc](https://github.com/antoninschrab/agginc/): Efficient MMDAggInc HSICAggInc KSDAggInc tests
- [mmdfuse](https://github.com/antoninschrab/mmdfuse/): MMD-Fuse test
- [dckernel](https://github.com/antoninschrab/dckernel/): Robust to Data Corruption dcMMD dcHSIC tests
