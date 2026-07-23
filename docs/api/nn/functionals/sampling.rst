Sampling Functionals
====================

Sample Without Replacement
--------------------------

.. autofunction:: physicsnemo.nn.functional.sample_without_replacement

The default ``exact`` strategy provides uniform or weighted sampling without
the :math:`2^{24}` category limit imposed by ``torch.multinomial``. The
``poisson_gap`` strategy is an explicit, low-memory approximation for
unweighted, ordered coverage of very large populations.
