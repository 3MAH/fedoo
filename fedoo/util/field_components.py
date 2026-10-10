"""Predefined component labels shared by datasets and the result viewer."""

FIELD_COMPONENTS = {
    "Stress": ("XX", "YY", "ZZ", "XY", "XZ", "YZ"),
    "Strain": ("XX", "YY", "ZZ", "XY", "XZ", "YZ"),
    "Disp": ("X", "Y", "Z"),
    "Rot": ("X", "Y", "Z"),
    "BeamStress": ("N", "QY", "QZ", "MX", "MY", "MZ"),
    "BeamStrain": ("epsX", "gammaY", "gammaZ", "kappaX", "kappaY", "kappaZ"),
    "BeamLocalFrame": ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"),
    "ShellLocalFrame": ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"),
    "ShellStress": ("NXX", "NYY", "NXY", "MXX", "MYY", "MXY", "QXZ", "QYZ"),
    "ShellStrain": (
        "epsXX",
        "epsYY",
        "gammaXY",
        "kappaXX",
        "kappaYY",
        "kappaXY",
        "gammaXZ",
        "gammaYZ",
    ),
}
for _name in ("PK2", "PKII", "Kirchhoff", "Cauchy"):
    FIELD_COMPONENTS[_name] = FIELD_COMPONENTS["Stress"]
for _name in ("Stress", "Strain"):
    for _suffix in ("_local", "_global"):
        FIELD_COMPONENTS[_name + _suffix] = FIELD_COMPONENTS[_name]


def get_field_components(field, n_components):
    """Labels for stored and derived components; numeric fallback for user fields."""
    field = field.removesuffix("_local").removesuffix("_global")
    if field == "Rot" and n_components == 1:
        labels = ["Z"]
    elif field in FIELD_COMPONENTS and (
        len(FIELD_COMPONENTS[field]) == n_components
        or field in ("Disp", "Rot")
        and n_components <= 3
    ):
        labels = list(FIELD_COMPONENTS[field][:n_components])
    else:
        return [str(i) for i in range(n_components)]
    if field in ("Stress", "PK2", "PKII", "Kirchhoff", "Cauchy"):
        labels += ["vm", "pressure", "I", "II", "III"]
    elif field == "Strain":
        labels += ["I", "II", "III"]
    elif field in ("Disp", "Rot"):
        labels += ["norm"]
    return labels


def stored_component_index(field, component, n_components):
    if field == "Rot" and n_components == 1 and component == "Z":
        return 0
    labels = FIELD_COMPONENTS.get(field, ())
    if component in labels and (
        len(labels) == n_components or field in ("Disp", "Rot")
    ):
        return labels.index(component)
    return component
