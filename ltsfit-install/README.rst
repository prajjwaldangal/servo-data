The LtsFit Package
==================

**Robust Linear Regression with Scatter in One or Two Dimensions**

.. image:: https://img.shields.io/pypi/v/ltsfit.svg
        :target: https://pypi.org/project/ltsfit/
.. image:: https://img.shields.io/badge/arXiv-1208.3522-orange.svg
    :target: https://arxiv.org/abs/1208.3522
.. image:: https://img.shields.io/badge/DOI-10.1093/mnras/stt562-green.svg
        :target: https://doi.org/10.1093/mnras/stt562

LtsFit is a Python implementation of the method described in Section 3.2 of
`Cappellari et al. (2013a) <https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1709C>`_
to perform **very robust** fits of lines or planes to data with errors in all
coordinates, while allowing for possible intrinsic scatter.
Outliers are iteratively clipped using the robust Least Trimmed Squares (LTS)
technique by `Rousseeuw & van Driessen (2006)
<http://dx.doi.org/10.1007/s10618-005-0024-4>`_.

Attribution
-----------

If you use this software for your research, please cite
`Cappellari et al. (2013a) <https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1709C>`_
where the implementation was described. The BibTeX entry for the paper is::

    @ARTICLE{Cappellari2013a,
        author = {{Cappellari}, M. and {Scott}, N. and {Alatalo}, K. and
            {Blitz}, L. and {Bois}, M. and {Bournaud}, F. and {Bureau}, M. and
            {Crocker}, A.~F. and {Davies}, R.~L. and {Davis}, T.~A. and {de Zeeuw},
            P.~T. and {Duc}, P.-A. and {Emsellem}, E. and {Khochfar}, S. and
            {Krajnovi{\'c}}, D. and {Kuntschner}, H. and {McDermid}, R.~M. and
            {Morganti}, R. and {Naab}, T. and {Oosterloo}, T. and {Sarzi}, M. and
            {Serra}, P. and {Weijmans}, A.-M. and {Young}, L.~M.},
        title = "{The ATLAS$^{3D}$ project - XV. Benchmark for early-type
            galaxies scaling relations from 260 dynamical models: mass-to-light
            ratio, dark matter, Fundamental Plane and Mass Plane}",
        journal = {MNRAS},
        eprint = {1208.3522},
        year = 2013,
        volume = 432,
        pages = {1709-1741},
        doi = {10.1093/mnras/stt562}
    }

Installation
------------

install with::

    pip install ltsfit

Without writing access to the global ``site-packages`` directory, use::

    pip install --user ltsfit

Documentation
-------------

See ``ltsfit/examples`` and the files headers.

License
-------

Copyright (c) 2012-2020 Michele Cappellari

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included in all
copies of the software. All other rights are reserved.
In particular, redistribution of the code is not allowed.

