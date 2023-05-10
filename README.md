# Welcome to sequence-modelling's documentation!

Numerically optimized time-series and sequence modelling in Python.

## Key features

- Hidden Markov Models and Quasi-Deterministic Hidden Markov Models
- Numerically stable: floating point arithmetic performed in log space to avoid underflow
- Easy to use (based on the scikit-learn API)
- Pure Python and Numpy based
- Open source and commercially usable (BSD license)
- Support for discrete and continuous emissions

## Installation

The easiest way to install sequence-modelling is using pip:

```python
   pip install sequence-modelling
```

## Example usage

```python

   import numpy as np
   from sequence_modelling.emissions import Gaussian
   from sequence_modelling.hmm import StandardHMM
   import sequence_modelling.hmmviz as plt

   # Build a 2-state HMM model with one-dimensional Gaussian emissions

   # the transition matrix
   A = np.array([[0.6, 0.4],
                 [0.3, 0.7],
                 [0.5, 0.5]])

   # the emission object
   O = Gaussian(mu=np.array([[-100.0, 100.0]]),
             covar=np.array([[[10.0]], [[10.0]]]))

   # Build the HMM model object
   hmm = StandardHMM(A, O)

   # Sample from the generative model
   obs, zes = hmm.sample(dim=2, N=1000)

    # Fit the model to the data
   likelihood, ll, duration, rankn, res = hmm.fit(obs)

   # Decode (Predict) the most likely state sequence using the Viterbi algorithm
   decoded_path = hmm.viterbi(obs)

   # Visualize the state sequence
   plt.plot_state_sequence(obs, decoded_path, hmm.O.mu, hmm.O.covar)
```
