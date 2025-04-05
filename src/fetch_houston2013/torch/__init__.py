import warnings
warnings.warn(
    "The 'torch dataset' feature is experimental and may change in future releases.",
    category=UserWarning,
    stacklevel=2
)

from fetch_houston2013 import fetch_houston2013, fetch_trento, fetch_muufl, split_spmatrix
from .common_hsi_dsm_dataset import CommonHsiDsmDataset

class Houston2013(CommonHsiDsmDataset):
    def __init__(self, subset, patch_size=5, *args, **kwargs):
        super().__init__(fetch_houston2013, subset, patch_size, *args, **kwargs)

class Muufl(CommonHsiDsmDataset):
    """
    
    This dataset is an opinionated version of the MUUFL Gulfport dataset. If you want to use the original dataset, please use `fetch_muufl`. Which tries to keep the original infomation.
    """
    def __init__(self, subset, patch_size=5, n_train_perclass=20, *args, **kwargs):
        def get_data():
            hsi, dsm, truth, info = fetch_muufl()
            train_truth, test_truth = split_spmatrix(truth, n_train_perclass)
            return hsi, dsm[0:1], train_truth, test_truth, info
        super().__init__(get_data, subset, patch_size, *args, **kwargs)

class Trento(CommonHsiDsmDataset):
    def __init__(self, subset, patch_size=5, n_train_perclass=20, *args, **kwargs):
        def get_data():
            hsi, dsm, truth, info = fetch_trento()
            train_truth, test_truth = split_spmatrix(truth, n_train_perclass)
            return hsi, dsm[0:1], train_truth, test_truth, info
        super().__init__(get_data, subset, patch_size, *args, **kwargs)



__all__ = ['Houston2013', 'Muufl', 'Trento']