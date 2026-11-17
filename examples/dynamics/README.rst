Dynamics and rigid bodies
-------------------------

Implicit time integration of structural dynamics (Newmark schemes attached to
the problem), dynamic crack propagation, and rigid bodies
(``fedoo.constraint.RigidBody``): free fall, then bodies of increasing
geometric complexity (sphere, Stanford bunny, cow) bouncing on a floor through
IPC contact, and a deformable shell ball dropped on a plate.

The examples with contact require the optional dependency ``ipctk``
(``pip install fedoo[ipc]``).
