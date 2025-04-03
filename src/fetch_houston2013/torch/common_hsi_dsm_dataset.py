# Assume that torch exists
from typing import Callable, Literal

from scipy.sparse import coo_array
from torchvision.datasets.vision import VisionDataset

import numpy as np

class CommonHsiDsmDataset(VisionDataset):
    def __init__(self,
                 data_fetch: Callable[[],tuple[np.ndarray,np.ndarray,coo_array,coo_array, dict]],
                 subset: Literal['train', 'test', 'full'], patch_radius: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_radius = patch_radius
        self.subset = subset

        self.HSI, self.LIDAR, train_truth, test_truth, self.INFO = data_fetch()
        self.n_class = self.INFO['n_class']

        # Preprocess HSI and DSM
        pad_shape = ((0, 0), (self.patch_radius, self.patch_radius), (self.patch_radius, self.patch_radius))
        self.hsi = np.pad(self.HSI,   pad_shape, 'symmetric')
        self.dsm = np.pad(self.LIDAR, pad_shape, 'symmetric')

        # Preprocess truth
        self.truth = {
            'train': train_truth,
            'test': test_truth,
            'full': coo_array(-1*np.ones_like(train_truth.todense(), dtype=np.int16), dtype='int')
        }[subset]


    def __len__(self):
        return self.truth.count_nonzero()

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        w = 2*self.patch_radius+1

        i = self.truth.row[index]
        j = self.truth.col[index]
        cid = self.truth.data[index].item()

        x_hsi = self.hsi[:, i:i+w, j:j+w]
        x_dsm = self.dsm[:, i:i+w, j:j+w]
        y = np.eye(self.n_class)[cid-1]

        # 额外信息: 当前点的坐标
        extras = {
            "location": [i, j],
            "index": index,
            "class": cid
        }

        return x_hsi, x_dsm, y, extras
