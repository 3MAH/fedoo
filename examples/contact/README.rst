Contact and interfaces
----------------------

Contact between deformable bodies and self-contact, with the penalty method and
with the Incremental Potential Contact (IPC) barrier, compared with analytical
solutions (Hertz, Westergaard) where they exist. The folder also covers
self-contact in 2D axisymmetric models (tube compression) and cohesive
interfaces: :ref:`sphx_glr_examples_contact_cohesive_dcb_3d.py` opens a
zero-thickness interface governed by a :class:`fedoo.constitutivelaw.CohesiveLaw`.

The IPC examples (``ipc_*`` and the other scripts using ``IPCContact`` or
``IPCSelfContact``) require the optional dependency ``ipctk``
(``pip install fedoo[ipc]``). Some of them also need ``gmsh`` to mesh curved
geometries.
