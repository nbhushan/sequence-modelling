.. sequence modelling documentation master file, created by
   sphinx-quickstart on Wed Mar  1 14:11:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sequence-modelling's documentation!
==============================================

Numerically optimized time-series and sequence modelling in Python.

Key features
------------
- Hidden Markov Models and Quasi-Deterministic Hidden Markov Models
- Numerically stable: floating point arithmetic performed in log space to avoid underflow
- Easy to use (based on the scikit-learn API)
- Pure Python and Numpy based
- Open source and commercially usable (BSD license)
- Support for discrete and continuous emissions

Example usage
-------------

.. code-block:: python

   import numpy as np
   from sequence_modelling.emissions import Gaussian
   from sequence_modelling.hmm import StandardHMM
   import sequence_modelling.hmmviz as plt

   # Build a 3-state HMM model with two-dimensional Gaussian emissions
   # the transition matrix 
   A = np.array([[0.9, 0.1, 0.0],
                 [0.0, 0.9, 0.1],
                 [0.0, 0.0, 1.0]])
   # the emission object 
   O = Gaussian(mu = np.array([[0.0, 1.0, 2.0],
                               [0.0, 1.0, 2.0]]),
               covar = np.array([[0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.1]]))
       
   # Build the HMM model object
   hmm = StandardHMM(A, O)
   
   # Sample from the generative model
   obs, zes = hmm.sample(dim=2, N=1000)
    ...
    # Fit the model to the data
   likelihood, ll, duration, rankn, res = hmm.fit(obs)
   
   # Decode (Predict) the most likely state sequence using the Viterbi algorithm
   decoded_path = hmm.viterbi(obs)
   
   # Visualize the state sequence
   plt.plot_state_sequence(obs, decoded_path, hmm.O.mu, hmm.O.covar)


.. toctree::
   :maxdepth: 
   :caption: Modules:

   docs/sequence_modelling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
