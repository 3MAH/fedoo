"""Guard against double registration of the same BC object.

A RigidTie explicitly added by the user AND auto-registered by a
RigidBodyAssembly (constraint/rigid_body.py _register_global_dofs) used to
end up TWICE in pb.bc: its MPCs were then duplicated, which silently
corrupted the elimination (MatCB) and destroyed Newton convergence (the
force-controlled cap benchmark diverged erratically from increment 4 with a
perfectly consistent tangent). ListBC.add now ignores an object identical
(by identity) to one already registered — in either registration order.
"""

import numpy as np
import pytest

simcoon = pytest.importorskip("simcoon")

import fedoo as fd
from fedoo.constraint.rigid_body import RigidBodyAssembly
from fedoo.constraint.rigid_tie import RigidTie


def _make_problem(add_tie_explicitly):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")
    mat = fd.constitutivelaw.Simcoon("NEOHC", [3.0, 150.0], name="law")
    mat.set_density(1000.0)
    wf = fd.weakform.StressEquilibriumRI(mat, nlgeom="UL")
    asm_fe = fd.Assembly.create(wf, mesh, name="fe")
    tie = fd.constraint.RigidTie(mesh.find_nodes("Z", mesh.bounding_box.zmax))
    rb = RigidBodyAssembly(
        mass=0.01,
        inertia_tensor=1e-6 * np.eye(3),
        rigid_tie=tie,
        mesh=mesh,
        name="rb",
    )
    pb = fd.problem.NonLinear(asm_fe + rb)
    pb.set_time_integrator(
        fd.time.SECOND_ORDER, fd.time.Newmark(beta=0.3025, gamma=0.6)
    )
    if add_tie_explicitly:
        pb.bc.add(tie)  # would duplicate without the ListBC.add identity guard
    pb.bc.add("Dirichlet", mesh.find_nodes("Z", mesh.bounding_box.zmin), "Disp", 0)
    pb.dtime = 0.01
    pb.initialize()
    return pb, tie


@pytest.mark.parametrize("add_tie_explicitly", [True, False])
def test_rigid_tie_registered_once(add_tie_explicitly):
    pb, tie = _make_problem(add_tie_explicitly)
    ties = [b for b in pb.bc if isinstance(b, RigidTie)]
    assert len(ties) == 1
    assert ties[0] is tie


def test_double_add_same_object_is_ignored():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")
    tie = fd.constraint.RigidTie(mesh.find_nodes("Z", mesh.bounding_box.zmax))
    bcs = fd.core.boundary_conditions.ListBC()
    bcs.add(tie)
    bcs.add(tie)
    bcs.append(tie)  # append is the guarded chokepoint (add goes through it)
    bcs.extend([tie, tie])
    assert sum(1 for b in bcs if b is tie) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
