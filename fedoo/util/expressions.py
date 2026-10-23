"""Array expressions shared by mesh selection, datasets and the viewer.

``X``, ``Y``, ``Z`` and ``ID`` refer to nodes or element centers. With a
dataset, fields and component aliases such as ``Stress_XX`` are available.
``field("Stress_XX")`` and ``scalar("Fext(RigidDisp)")`` handle names that
cannot be Python identifiers. Named sets provide indices through
``nodes("Top")`` / ``elements("Wall")`` or ``node_sets["Top"]``.

Arithmetic, comparisons, array indexing, boolean operators and an explicit
set of NumPy functions are supported. Expressions do not execute Python
statements, arbitrary attributes, imports or user-provided callables.
"""

import ast
import io
import operator
import tokenize

import numpy as np


FUNCTIONS = {
    name: getattr(np, name)
    for name in (
        "abs",
        "sqrt",
        "exp",
        "log",
        "log10",
        "sin",
        "cos",
        "tan",
        "arctan2",
        "sinh",
        "cosh",
        "tanh",
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "var",
        "median",
        "ptp",
        "count_nonzero",
        "where",
        "clip",
        "minimum",
        "maximum",
        "isclose",
        "isfinite",
        "logical_and",
        "logical_or",
        "logical_not",
    )
}
FUNCTIONS["norm"] = np.linalg.norm
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_ARITY = {
    "abs": 1,
    "sqrt": 1,
    "exp": 1,
    "log": 1,
    "log10": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "isfinite": 1,
    "logical_not": 1,
    "arctan2": 2,
    "minimum": 2,
    "maximum": 2,
    "logical_and": 2,
    "logical_or": 2,
    "where": 3,
    "clip": 3,
    "isclose": 2,
    "sum": 2,
    "mean": 2,
    "min": 2,
    "max": 2,
    "std": 2,
    "var": 2,
    "median": 2,
    "ptp": 2,
    "count_nonzero": 2,
    "norm": 1,
    "nodes": 1,
    "elements": 1,
    "scalar": 1,
    "field": 2,
}


def mesh_positions(mesh, data_type="Node"):
    if data_type == "Node":
        return mesh.nodes
    if data_type == "Element":
        blocks = getattr(mesh, "submeshes", (mesh,))
        return np.concatenate([block.element_centers for block in blocks], axis=0)
    raise ValueError("Expression association must be Node or Element")


def nodes_to_elements(mesh, indices, all_nodes=True):
    """Convert physical node ids to global element ids, across all submeshes."""
    indices = np.asarray(list(indices), dtype=int)
    if np.any((indices < 0) | (indices >= mesh.n_nodes)):
        raise ValueError("Node ID outside the mesh")
    blocks = getattr(mesh, "submeshes", (mesh,))
    result, offset = [], 0
    for block in blocks:
        included = np.isin(block.elements, indices)
        mask = included.all(axis=1) if all_nodes else included.any(axis=1)
        result.extend(np.flatnonzero(mask) + offset)
        offset += block.n_elements
    return np.asarray(result, dtype=int)


def elements_to_nodes(mesh, indices):
    """Return unique physical nodes used by global element ids."""
    indices = np.asarray(list(indices), dtype=int)
    if np.any((indices < 0) | (indices >= mesh.n_elements)):
        raise ValueError("Element ID outside the mesh")
    result, offset = [], 0
    for block in getattr(mesh, "submeshes", (mesh,)):
        local = (
            indices[(indices >= offset) & (indices < offset + block.n_elements)]
            - offset
        )
        result.extend(block.elements[local].ravel())
        offset += block.n_elements
    return np.unique(np.asarray(result, dtype=int))


def expression_field(data, name, data_type="Node"):
    """Resolve an exact field name before interpreting a component suffix."""
    if data is None:
        raise ValueError("Field expressions require a dataset")
    derived = data.section_derived_fields()
    names = set(data.field_names())
    names.update(description[0] for description in derived.values())
    component = None
    if name in names:
        field = name
    else:
        matches = [field for field in names if name.startswith(field + "_")]
        if not matches:
            raise ValueError(f"Unknown field or component: {name}")
        field = max(matches, key=len)
        component = name[len(field) + 1 :]
        if component.isdigit():
            component = int(component)
    if field + "_local" in derived:
        field += "_local"
    values = data.get_data(field, component, data_type, fill_unused_nodes=None)
    if hasattr(values, "to_global"):
        values = values.to_global(fill_value=np.nan)
    return np.asarray(values)


def evaluate_expression(
    expression,
    *,
    mesh=None,
    data=None,
    data_type="Node",
    positions=None,
    scalar_resolver=None,
):
    """Evaluate a mathematical expression using mesh positions and dataset fields.

    Bare fields use ``data_type``; ``field("Stress_XX", "Element")`` can
    explicitly request another association. ``and``/``or`` and ``&``/``|``
    operate elementwise, including the legacy unparenthesized mesh syntax.
    NumPy functions may be written as ``sum(...)`` or ``np.sum(...)``.
    """
    mesh = mesh if mesh is not None else getattr(data, "mesh", None)
    if positions is None and mesh is not None:
        positions = mesh_positions(mesh, data_type)
    environment = {"pi": np.pi, "e": np.e, "nan": np.nan}
    if positions is not None:
        positions = np.asarray(positions)
        padded = np.pad(positions, ((0, 0), (0, max(0, 3 - positions.shape[1]))))
        environment.update(zip(("X", "Y", "Z"), padded.T[:3]))
        environment["ID"] = np.arange(len(positions))
    node_sets = {} if mesh is None else dict(mesh.node_sets)
    element_sets = {} if mesh is None else dict(mesh.element_sets)
    if mesh is not None:
        node_sets.setdefault("All", np.arange(mesh.n_nodes))
        element_sets.setdefault("All", np.arange(mesh.n_elements))
    node_sets = {name: np.asarray(ids, dtype=int) for name, ids in node_sets.items()}
    element_sets = {
        name: np.asarray(ids, dtype=int) for name, ids in element_sets.items()
    }
    environment.update(node_sets=node_sets, element_sets=element_sets)

    def scalar(name):
        if scalar_resolver is not None:
            return scalar_resolver(name)
        if data is None or name not in data.scalar_data:
            raise ValueError(f"Unknown scalar: {name}")
        return data.scalar_data[name]

    functions = dict(FUNCTIONS)
    functions.update(
        nodes=lambda name: node_sets[name],
        elements=lambda name: element_sets[name],
        scalar=scalar,
        field=lambda name, association=data_type: expression_field(
            data, name, association
        ),
    )
    tokens = tokenize.generate_tokens(io.StringIO(expression).readline)
    try:
        normalized = tokenize.untokenize(
            (tokenize.NAME, "and" if token.string == "&" else "or")
            if token.type == tokenize.OP and token.string in ("&", "|")
            else (token.type, token.string)
            for token in tokens
        )
        tree = ast.parse(normalized.strip(), mode="eval")
    except (SyntaxError, tokenize.TokenError) as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc

    def visit(node):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                raise ValueError("Unsupported constant")
            return node.value
        if isinstance(node, ast.Name):
            if node.id in environment:
                return environment[node.id]
            if data is not None:
                if node.id in data.scalar_data or (
                    scalar_resolver is not None
                    and node.id in getattr(data, "scalar_expressions", {})
                ):
                    return scalar(node.id)
                return expression_field(data, node.id, data_type)
            raise ValueError(f"Unknown name: {node.id}")
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [visit(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Slice):
            return slice(
                *(
                    visit(item) if item is not None else None
                    for item in (node.lower, node.upper, node.step)
                )
            )
        if isinstance(node, ast.Subscript):
            return visit(node.value)[visit(node.slice)]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return np.logical_not(value)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
        if isinstance(node, ast.BoolOp):
            function = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
            result = visit(node.values[0])
            for item in node.values[1:]:
                result = function(result, visit(item))
            return result
        if isinstance(node, ast.Compare):
            left, result = visit(node.left), True
            for comparison, item in zip(node.ops, node.comparators):
                if type(comparison) not in _COMPARE:
                    raise ValueError("Unsupported comparison")
                right = visit(item)
                result = np.logical_and(result, _COMPARE[type(comparison)](left, right))
                left = right
            return result
        if isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "np"
            ):
                name = function.attr
                allowed = FUNCTIONS
            elif isinstance(function, ast.Name):
                name = function.id
                allowed = functions
            else:
                raise ValueError("Only named mathematical functions are allowed")
            if name not in allowed:
                raise ValueError(f"Unsupported function: {name}")
            if len(node.args) > _ARITY[name]:
                raise ValueError(f"Too many positional arguments for {name}")
            if any(
                keyword.arg not in {"axis", "keepdims", "ddof", "rtol", "atol"}
                for keyword in node.keywords
            ):
                raise ValueError("Unsupported function keyword")
            return allowed[name](
                *(visit(argument) for argument in node.args),
                **{keyword.arg: visit(keyword.value) for keyword in node.keywords},
            )
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")

    try:
        with np.errstate(all="ignore"):
            return visit(tree.body)
    except (KeyError, IndexError, TypeError, NameError, ArithmeticError) as exc:
        raise ValueError(f"Cannot evaluate expression: {exc}") from exc


def selection_mask(
    expression, *, mesh=None, data=None, data_type="Node", positions=None
):
    mesh = mesh if mesh is not None else data.mesh
    count = mesh.n_nodes if data_type == "Node" else mesh.n_elements
    result = np.asarray(
        evaluate_expression(
            expression, mesh=mesh, data=data, data_type=data_type, positions=positions
        )
    )
    if result.dtype != bool or result.shape not in ((), (count,)):
        raise ValueError(f"Selection must return one boolean per {data_type.lower()}")
    return np.broadcast_to(result, (count,))


def evaluate_scalar_expressions(data, definitions=None):
    """Evaluate dependent scalar definitions once per frame, detecting cycles."""
    definitions = data.scalar_expressions if definitions is None else definitions
    values, visiting = {}, set()

    def resolve(name):
        if name in values:
            return values[name]
        if name not in definitions:
            if name not in data.scalar_data:
                raise ValueError(f"Unknown scalar: {name}")
            return data.scalar_data[name]
        if name in visiting:
            raise ValueError(f"Cyclic scalar expression: {name}")
        visiting.add(name)
        definition = definitions[name]
        value = np.asarray(
            evaluate_expression(
                definition["expression"],
                data=data,
                data_type=definition["data_type"],
                scalar_resolver=resolve,
            )
        )
        if value.ndim != 0 or value.dtype.kind not in "biuf" or not np.isfinite(value):
            raise ValueError(f"'{name}' must evaluate to one finite numeric scalar")
        visiting.remove(name)
        values[name] = value.item()
        return values[name]

    for name in definitions:
        resolve(name)
    return values
