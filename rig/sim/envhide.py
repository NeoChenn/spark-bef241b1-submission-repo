LIGHT_TYPES = {"DomeLight", "DistantLight"}
SEA_PREFIXES = ("sea", "ocean", "water")


def is_builtin_environment(name: str, type_name: str) -> bool:
    if type_name in LIGHT_TYPES:
        return True
    low = name.lower()
    return type_name == "Mesh" and any(low == p or low.startswith(p + "_") for p in SEA_PREFIXES)
