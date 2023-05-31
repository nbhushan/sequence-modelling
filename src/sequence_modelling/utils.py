# -*- coding: utf-8 -*-
"""
Numerical stability helpers

@author: nbhushan

"""
import numpy as np


def logsumexp(a, axis=None):
    """Compute the log of the sum of exponentials of input elements,
       Modified scipy.misc logsumexp to accept [-inf,-inf].

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which the sum is taken. By default `axis` is None,
        and all elements are summed.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    Numpy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    Examples
    --------
    >>> from scipy.misc import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107

    """
    a = np.asarray(a)
    if axis is None:
        a = a.ravel()
        a_max = a.max()
        if np.isinf(a_max):
            return a_max
        else:
            return a_max + np.log(np.sum(np.exp(a - a_max)))
    else:
        a = np.rollaxis(a, axis)
        a_max = a.max(axis=0)
        indices = np.isinf(a_max)
        a_max[indices] = 0.0
        out = np.log(np.sum(np.exp(a - a_max), axis=0))
        out += a_max
        return out
