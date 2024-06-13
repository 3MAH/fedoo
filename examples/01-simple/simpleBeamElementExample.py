"""
Beam Element Canteleaver Beam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple canteleaver beam example using different kind of elements
"""

import fedoo as fd
import numpy as np

E = 1e5
nu = 0.3

fd.ModelingSpace("3D")
fd.constitutivelaw.ElasticIsotrop(E, nu, name="ElasticLaw")

# circular section
R = 1
Section = np.pi * R**2
Jx = np.pi * R**4 / 2
Iyy = np.pi * R**4 / 4
Izz = np.pi * R**4 / 4
k = 0.8  # reduce section for shear (k=0 -> no shear effect)

L = 10  # beam lenght
F = -2  # Force applied on right section

# Build a straight beam mesh
Nb_elm = 10  # Number of elements
crd = np.linspace(0, L, Nb_elm + 1).reshape(-1, 1) * np.array([[1, 0, 0]])
# crd = np.linspace(0,L,Nb_elm+1).reshape(-1,1)* np.array([[0,0,1]]) #beam oriented in the Z axis
elm = np.c_[np.arange(0, Nb_elm), np.arange(1, Nb_elm + 1)]

fd.Mesh(crd, elm, "lin2", name="beam")
nodes_left = [0]
nodes_right = [Nb_elm]

# computeShear = 0: no shear strain are considered. Bernoulli element is used ("i.e "bernoulliBeam" element)
# computeShear = 1: shear strain using the "beam" element (shape functions depend on the beam parameter) ->  Friedman, Z. and Kosmatka, J. B. (1993).  An improved two-node Timoshenkobeam finite element.Computers & Structures, 47(3):473–481
# computeShear = 2: shear strain using the "beamFCQ" element (using internal variables) -> Caillerie, D., Kotronis, P., and Cybulski, R. (2015). A new Timoshenko finite element beamwith internal degrees of freedom.International Journal of Numerical and Analytical Methods in Geomechanics
computeShear = 1

if computeShear == 0:
    fd.weakform.BeamEquilibrium(
        "ElasticLaw", Section, Jx, Iyy, Izz, name="WFbeam"
    )  # by default k=0 i.e. no shear effect
    fd.Assembly.create("WFbeam", "beam", "bernoulliBeam", name="beam")
elif computeShear == 1:
    fd.weakform.BeamEquilibrium("ElasticLaw", Section, Jx, Iyy, Izz, k=k, name="WFbeam")
    fd.Assembly.create("WFbeam", "beam", "beam", name="beam")
else:  # computeShear = 2
    fd.Mesh["beam"].add_internal_nodes(
        1
    )  # adding one internal nodes per element (this node has no geometrical sense)
    fd.weakform.Beam("ElasticLaw", Section, Jx, Iyy, Izz, k=k, name="WFbeam")
    fd.Assembly.create("WFbeam", "beam", "beamFCQ", name="beam")

pb = fd.problem.Linear("beam")

pb.bc.add("Dirichlet", nodes_left, ["Disp", "Rot"], 0)
pb.bc.add("Neumann", nodes_right, "DispY", F)

pb.solve()

# Post treatment
results = pb.get_results(fd.Assembly["beam"], ["Fext"])["Fext"]

print("Reaction RX at the clamped extermity: " + str(results[0][0]))
print("Reaction RY at the clamped extermity: " + str(results[1][0]))
print("Reaction RZ at the clamped extermity: " + str(results[2][0]))
print("Moment MX at the clamped extermity: " + str(results[3][0]))
print("Moment MY at the clamped extermity: " + str(results[4][0]))
print("Moment MZ at the clamped extermity: " + str(results[5][0]))

print("RX at the free extremity: " + str(results[0][nodes_right[0]]))
print("RZ at the free extremity: " + str(results[2][nodes_right[0]]))

results = pb.get_results("beam", "BeamStress")["BeamStress"]
IntMoment = np.array(results[3:])
IntForce = np.array(results[:3])

U = np.reshape(pb.get_dof_solution("all"), (6, -1)).T
Theta = U[: nodes_right[0] + 1, 3:]
U = U[: nodes_right[0] + 1, 0:3]

sol = F * L**3 / (3 * E * Izz)
if computeShear != 0 and k != 0:
    G = E / (1 + nu) / 2
    sol += F * L / (k * G * Section)
print("Analytical deflection: ", sol)
print(U[-1])

M = fd.Assembly["beam"].global_matrix.todense()
