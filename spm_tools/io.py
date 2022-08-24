import pathlib
import warnings

import numpy as np

from ._uniform_coords import uniform_XYZ_decorator


def import_data(path, filetype=None):
    path = pathlib.Path(path)

    if filetype is None:
        filetype = path.suffix[1:].lower()

    return _import_method_dict[filetype](path)


@uniform_XYZ_decorator
def export_data(X, Y, Z, path, filetype=None, **kwargs):
    path = pathlib.Path(path)

    if filetype is None:
        filetype = path.suffix[1:].lower()

    return _export_method_dict[filetype](X, Y, Z, path, **kwargs)


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

    remaining_buffer = b"".join(lines[1 + n_metadata_lines :])
    nul_bites = ((len(remaining_buffer) - 1) % 4) + 1

    # Normalised range 0 to 1 describes edges of pixels, not centres
    x = np.linspace(
        1 / (2 * metadata["XRes"]), 1 - 1 / (2 * metadata["XRes"]), metadata["XRes"]
    )
    y = np.linspace(
        1 / (2 * metadata["YRes"]), 1 - 1 / (2 * metadata["YRes"]), metadata["YRes"]
    )

    # Rescale normalised coordinates
    if "XReal" in metadata:
        x *= metadata["XReal"]
    if "YReal" in metadata:
        y *= metadata["YReal"]
    if "XOffset" in metadata:
        x += metadata["XOffset"]
    if "YOffset" in metadata:
        y += metadata["YOffset"]

    X, Y = np.meshgrid(x, y)
    Z = np.frombuffer(remaining_buffer[nul_bites:], dtype=np.float32).reshape(
        metadata["YRes"], metadata["XRes"]
    )
    assert X.shape == Y.shape == Z.shape

    return X, Y, Z, metadata


def _export_gsf(X, Y, Z, path, metadata=None):
    if metadata is None:
        metadata = {}
        metadata_updates = {}
    else:
        # Check for consistency between metadata and given data
        if "XRes" in metadata:
            if Z.shape[1] != metadata["XRes"]:
                raise ValueError("Supplied 'XRes' must match data shape.")
        if "YRes" in metadata:
            if Z.shape[0] != metadata["YRes"]:
                raise ValueError("Supplied 'YRes' must match data shape.")

        # Any of these keys will override the data coordinates of `X` and `Y`
        metadata_overrides = "XReal", "YReal", "XOffset", "YOffset"
        metadata_updates = {
            key: metadata[key] for key in metadata_overrides if key in metadata
        }
        if metadata_updates:
            warnings.warn(
                "Metadata coordinates will override given `X` and `Y` values."
            )

    # Calculate metadata coordinates from data
    metadata["XRes"], metadata["YRes"] = Z.shape[::-1]
    dx = X.ptp() / metadata["XRes"]
    dy = Y.ptp() / metadata["YRes"]
    metadata["XReal"] = X.ptp() + dx
    metadata["YReal"] = Y.ptp() + dy
    metadata["XOffset"] = X.min() - dx / 2
    metadata["YOffset"] = Y.min() - dy / 2

    # Apply any overrides
    metadata.update(metadata_updates)

    header = "Gwyddion Simple Field 1.0\n"
    first_headers = "XRes", "YRes"
    for key in first_headers:
        header += f"{key} = {metadata[key]}\n"
    for key, val in metadata.items():
        if key not in first_headers:
            header += f"{key} = {val}\n"

    with open(path, "wb") as file:
        file.write(bytes(header, "utf-8"))
        file.write(b"\x00" * (4 - len(header) % 4))
        file.write(Z.tobytes())


_import_method_dict = {
    "gsf": _import_gsf,
}

_export_method_dict = {
    "gsf": _export_gsf,
}
