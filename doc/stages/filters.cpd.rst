.. _filters.cpd:

filters.cpd
==============

The CPD filter uses the Coherent Point Drift :cite:`Myronenko` algorithm to
compute a rigid, nonrigid, or affine transformation between datasets.  The
rigid and affine are what you'd expect; the nonrigid transformation uses Motion
Coherence Theory :cite:`Yuille1998` to "bend" the points to find a best
alignment.

.. note::

    CPD is computationally intensive and can be slow when working with many
    points (i.e. :math:`> 10,000`).  Nonrigid is significatly slower than rigid and
    affine.

The first input to the change filter are considered the "fixed" points, and all
subsequent inputs are "moving" points.  The output from the change filter are
the "moving" points after the calculated transformation has been applied, one
point view per input.  Any additional information about the cpd registration,
e.g. the rigid transformation matrix, will be placed in the stage's metadata.

.. plugin::

Examples
--------

.. code-block:: json

    {
        "pipeline": [
            "fixed.las",
            "moving.las",
            {
                "type": "filters.cpd",
                "method": "rigid"
            },
            "output.las"
        ]
    }

If "method" is not provided, the cpd filter will default to using the rigid registration method.
To get the transform matrix, you'll need to use the ``--metadata`` option:

::

    $ pdal pipeline cpd-pipeline.json --metadata cpd-metadata.json

The metadata output might start something like:

.. code-block:: json

    {
        "stages":
        {
            "filters.cpd":
            {
                "iterations": 10,
                "method": "rigid",
                "runtime": 0.003839,
                "sigma2": 5.684342128e-16,
                "transform": "           1 -6.21722e-17  1.30104e-18  5.29303e-11-8.99346e-17            1  2.60209e-18 -3.49247e-10 -2.1684e-19  1.73472e-18            1 -1.53477e-12           0            0            0            1"
            },
        },

.. seealso::

    :ref:`filters.transformation` to apply a transform to other points.

Options
--------

method
    Change detection method to use.
    Valid values are "rigid", "affine", and "nonrigid".
    [Default: **rigid**]

.. _Coherent Point Drift (CPD): https://github.com/gadomski/cpd

.. bibliography:: references.bib
