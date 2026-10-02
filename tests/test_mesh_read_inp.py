import fedoo as fd
import numpy as np
import pytest

INP = """*Heading
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
9, 0.5, 0.5, 2.0
*ELEMENT, type=C3D8, ELSET=Hexa
1, 1, 2, 3, 4, 5, 6, 7, 8
*ELEMENT, type=C3D4, ELSET=Tetra
2, 5, 6, 7, 9
3, 5, 7, 8, 9
*NSET, NSET=Bottom
1, 2, 3, 4
*ELSET, ELSET=LastTet
3
"""


@pytest.fixture
def inp_file(tmp_path):
    path = tmp_path / "mixed.inp"
    path.write_text(INP)
    return str(path)


def _check(mesh):
    assert isinstance(mesh, fd.MultiMesh)
    assert mesh.nodes.shape == (9, 3)
    assert set(mesh.mesh_dict) == {"hex8", "tet4"}
    assert mesh["hex8"].elements.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert mesh["tet4"].elements.tolist() == [[4, 5, 6, 8], [4, 6, 7, 8]]
    assert np.array_equal(np.sort(mesh.node_sets["Bottom"]), [0, 1, 2, 3])
    assert mesh["hex8"].element_sets["Hexa"].tolist() == [0]
    assert np.sort(mesh["tet4"].element_sets["Tetra"]).tolist() == [0, 1]
    assert mesh["tet4"].element_sets["LastTet"].tolist() == [1]


@pytest.mark.parametrize("library", ["meshlane", "meshio"])
def test_from_meshio(inp_file, library):
    io = pytest.importorskip(library)
    _check(fd.Mesh.from_meshio(io.read(inp_file)))


def test_read_inp(inp_file):
    if not fd.core.mesh.USE_MESHIO:
        pytest.skip("neither meshlane nor meshio is installed")
    _check(fd.Mesh.read(inp_file))
