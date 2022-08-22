import pathlib
import numpy as np


def _import_gsf(path):
    with open(path, "rb") as file:
        lines = list(file)

    # Check header
    if lines[0] != b"Gwyddion Simple Field 1.0\n":
        raise ValueError("file has incorrect header for gsf")

    # Get metadata
    metadata = {}
    for n_metadata_lines, line in enumerate(lines[1:]):
        if line[:1] == b"\x00":
            break
        key, val = [s.strip() for s in line.decode().split("=")]
        metadata[key] = val

    # Convert known metadata values from strings
    for key in ("XRes", "YRes"):
        if key in metadata:
            metadata[key] = np.int(metadata[key])
        else:
            raise ValueError(f"gsf file has no value for {key}")
    for key in ("XReal", "YReal", "XOffset", "YOffset"):
        if key in metadata:
            metadata[key] = np.float(metadata[key])

    Z = np.frombuffer(
        b"".join(lines[1 + n_metadata_lines :])[4:], dtype=np.float32
    ).reshape(metadata["YRes"], metadata["XRes"])

    x = np.linspace(0, 1, metadata["XRes"])
    y = np.linspace(0, 1, metadata["YRes"])

    if "XReal" in metadata:
        x *= metadata["XReal"]
    if "YReal" in metadata:
        y *= metadata["YReal"]
    if "XOffset" in metadata:
        x += metadata["XOffset"]
    if "YOffset" in metadata:
        y += metadata["YOffset"]

    X, Y = np.meshgrid(x, y)
    return X, Y, Z, metadata


_import_method_dict = {
    "gsf": _import_gsf,
}


def import_data(path, filetype=None):
    path = pathlib.Path(path)

    if filetype is None:
        filetype = path.suffix[1:].lower()

    return _import_method_dict[filetype](path)
