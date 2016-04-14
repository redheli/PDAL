.. _reprojection:

Reprojection
================================================================================

.. index:: Reprojection, WGS84, UTM

.. include:: ../../includes/substitutions.rst

Exercise
--------------------------------------------------------------------------------

This exercise uses PDAL to reproject `ASPRS LAS`_ data

.. _`LASzip`: http://laszip.org
.. _`ASPRS LAS`: http://www.asprs.org/Committee-General/LASer-LAS-File-Format-Exchange-Activities.html

Issue the following command in your `Docker Quickstart Terminal`.


.. literalinclude:: ./reprojection-command-1.txt
    :linenos:

.. image:: ../../images/reprojection-run-command.png


Unfortunately this doesn't produce the intended results for us. Issue the following
``pdal info`` command to see why:

::

    docker run -v /c/Users/Howard/PDAL:/data -t pdal/pdal \
        pdal info /data/exercises/translation/csite-dd.laz --all

.. image:: ../../images/reprojection-wrong-scale.png


Some formats, like :ref:`writers.las` do not automatically set scaling
information. PDAL cannot really do this for you because there are a number
of ways to trip up. For latitude/longitude data, you will need to set
the scale to smaller values like ``0.0000001``. Additionally, LAS uses
an offset value to move the origin of the value. Use PDAL to set that
to ``auto`` so you don't have to compute it.

.. literalinclude:: ./reprojection-command-2.txt
    :linenos:

.. image:: ../../images/reprojection-run-with-scale.png


Run the `pdal info` command again to verify the ``X``, ``Y``, and ``Z``
dimensions:


.. image:: ../../images/reprojection-proper-scale.png

Notes
--------------------------------------------------------------------------------

1. :ref:`filters.reprojection` will use whatever coordinate system is defined
   by the point cloud file, but you can override it using the ``in_srs``
   option. This is useful in situations where the coordinate system
   is not correct, not completely specified, or your system doesn't have
   all of the required supporting coordinate system dictionaries.

2. PDAL uses `proj.4`_ library for reprojection. This library includes
   the capability to do both vertical and horizontal datum transformations.

.. _`proj.4`: http://proj4.org

.. _`CSV`: https://en.wikipedia.org/wiki/Comma-separated_values
.. _`JSON`: https://en.wikipedia.org/wiki/JSON
.. _`XML`: https://en.wikipedia.org/wiki/XML