import torch
from torch.utils.data import Dataset
import h5py

# Candidate columns holding the absolute feature-bag path (split_csv / protocols).
_PATH_COLS = ("patient_files", "tensor_paths", "patient_name")


class MILDataset(Dataset):
    """MIL dataset over a table of **absolute** feature-bag paths + integer labels.

    The unified split CSVs and the generated protocols both store absolute paths,
    so there is no directory to join -- a row's path is read straight from disk.
    """

    def __init__(self, data_files, ext="h5"):
        assert ext in ("pt", "h5"), f"Unsupported file extension: {ext}"
        self.df = data_files.reset_index(drop=True)
        self.ext = ext
        self.path_col = next((c for c in _PATH_COLS if c in self.df.columns), None)
        assert self.path_col is not None, f"no path column in {list(self.df.columns)}"

    def __len__(self):
        return len(self.df)

    def _path(self, idx):
        p = str(self.df[self.path_col][idx])
        return p if p.endswith(f".{self.ext}") else f"{p}.{self.ext}"

    def _read(self, path):
        features = read_ptfile(path) if self.ext == "pt" else read_h5file(path)
        if features.ndim == 3:            # (1, N, D) -> (N, D)
            features = features.squeeze(0)
        assert features.ndim == 2, f"Expected [N, D], got {features.shape}"
        return features

    def get_dimension(self):
        return 0 if len(self.df) == 0 else self._read(self._path(0)).shape[1]

    def __getitem__(self, idx):
        features = self._read(self._path(idx))
        label = self.df["labels"][idx]
        return features, label, ["dummy"], str(self.df[self.path_col][idx])


def read_h5file(file_name):
    with h5py.File(file_name, "r") as hf:
        return hf["features"][()]


def read_ptfile(file_name):
    return torch.load(file_name, map_location=torch.device("cpu"))["features"]
