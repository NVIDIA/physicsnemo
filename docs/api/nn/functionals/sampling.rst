Sampling Functionals
====================

Weighted Multinomial
--------------------

.. autofunction:: physicsnemo.nn.functional.weighted_multinomial

The core API follows ``torch.multinomial`` while also accepting an integer
uniform population size. Without replacement, the default ``exact`` strategy
provides uniform or weighted sampling without the :math:`2^{24}` category
limit imposed by ``torch.multinomial``. The ``poisson_gap`` strategy is an
explicit, low-memory approximation for unweighted, ordered coverage of very
large populations. Sampling with replacement is also supported.
