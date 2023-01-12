.. Symlearn documentation master file, created by
   sphinx-quickstart on Tue Jan 10 21:47:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Symlearn's Package documentation!
====================================

``symlearn`` is a python package that provides an API interface of different population-based optimization algorithms including `Firefly Algorithm`, `Immune Plasma Algorithm`, and others. ``symlearn`` uses the prencibles of symbolic regression to solve different regression problems that does not need any prespecified model. 

Symbolic regression is a method that is concerned with identifying the mathematical form that generates output variables using a set of independent inputs through a particular system. The process starts by searching in the space of the different functional expression spaces and combining them to form a mathematical model. Different regression methods have a predefined form that is used as an initial form to be optimized.

.. code-block:: python

    import math
    print 'import dones'

.. autoclass:: symlearn.core.node.Node

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
