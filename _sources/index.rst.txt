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

Installation
------------

The easiest way to install sequence-modelling is using pip:

.. code-block:: bash

   pip install sequence-modelling


Example usage
-------------

.. code-block:: python


   import numpy as np
   from sequence_modelling.emmissions import Gaussian
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

   # Sample 1000 observations from the generative model
   obs, path = hmm.sample(dim=1, N=100)

    # Fit the model to the data
   likelihood, ll, duration, rankn, res = hmm.hmmFit([obs])

   # Decode (Predict) the most likely state sequence using the Viterbi algorithm
   decoded_path = hmm.viterbi(obs)

   # Visualize the state sequence
   from matplotlib.pyplot import figure, show
   fa = figure()
   plt.view_viterbi(fa.add_subplot(1, 1, 1), [obs], [decoded_path], hmm.O.mu, seq=0)
   fa.tight_layout()
   show()


.. toctree::
    :hidden:
    :glob:

    *

   docs/sequence_modelling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
