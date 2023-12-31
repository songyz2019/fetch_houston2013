import os
from os.path import exists, expanduser, join
from pathlib import Path
import warnings
import hashlib
from zipfile import ZipFile
import urllib
import urllib.request
import logging

import numpy as np
import scipy.io

import skimage
from torch.utils.data import Dataset
from scipy.sparse import coo_array
import h5py


def fetch_muufl(datahome=None, download_if_missing=True):
    """
    Donwload and load the MUUFL Gulfport dataset.

    Use CHW format
    """
    def _verify_files(root: Path, files_sha256: dict, extra_message: str = '') -> None:
        """验证root下的文件的sha256是否与files_sha256相符

        :param extra_message: 额外报错信息
        :param files_sha256: 例如: `{"1.txt", "f4d619....", "2.txt": "9d03010....."}`
        :param root: 文件夹目录
        """

        def sha256(path):
            """Calculate the sha256 hash of the file at path."""
            sha256hash = hashlib.sha256()
            chunk_size = 8192
            with open(path, "rb") as f:
                while True:
                    buffer = f.read(chunk_size)
                    if not buffer:
                        break
                    sha256hash.update(buffer)
            return sha256hash.hexdigest()

        for filename, checksum in files_sha256.items():
            assert sha256(
                root / filename) == checksum, f"Incorrect SHA256 for {filename}. Expect {checksum}, Actual {sha256(root / filename)}. {extra_message}"

    def _get_data_home(data_home=None) -> str:
        if data_home is None:
            data_home = os.environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
        data_home = expanduser(data_home)
        os.makedirs(data_home, exist_ok=True)
        return data_home
    def fetch_zip(url, path: Path, download_if_missing: bool = True) -> Path:
        """Make sure `path` is the zip file of Houston2013, or raise FileNotFoundError
        """
        if not exists(path):
            if download_if_missing:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
                urllib.request.install_opener(opener)
                logger.info(f"Downloading {url}")
                urllib.request.urlretrieve(url, path)
            else:
                raise FileNotFoundError(f"{path} not found")

        _verify_files(path.parent, {path.name: "2219e6259e3ad80521a8a7ff879916624efa61eb6df1bfd80538f6f2d4befa2c"})
        return path

    # 1. 准备
    logger = logging.getLogger("fetch_muufl")
    URL = "https://github.com/GatorSense/MUUFLGulfport/archive/refs/tags/v0.1.zip"
    DATA_HOME = Path(_get_data_home(datahome))
    ZIP_PATH = DATA_HOME / 'MUUFLGulfport.zip'
    UNZIPED_PATH = DATA_HOME / 'MUUFLGulfport/'
    ROOT = UNZIPED_PATH/'MUUFLGulfport-0.1'
    FILES_SHA256 = {
        "MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat": "69420a72866dff4a858ae503e6e2981af46f406a4ad8f4dd642efa43feb59edc"
    }
    if not exists(DATA_HOME):
        os.makedirs(DATA_HOME)

    # 2.下载数据集zip文件并解压
    if exists(ROOT) and len(os.listdir(ROOT)) > 0:  # 已存在解压的文件夹且非空
        _verify_files(ROOT, FILES_SHA256, f"please try removing {ROOT}")
    else:
        fetch_zip(URL, ZIP_PATH, download_if_missing)
        # 解压 2013_DFTC 目录下的所有文件
        logger.info(f"Decompressing {ZIP_PATH}")
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)

        _verify_files(ROOT, FILES_SHA256)

        # 删除ZIP
        os.remove(ZIP_PATH)

        # 显示版权信息
        with open(ROOT / 'LICENSE', 'r', encoding='utf-8') as f:
            logger.info(f.read())

    # 3. 数据加载
    d = scipy.io.loadmat(
        ROOT / 'MUUFLGulfportSceneLabels' / 'muufl_gulfport_campus_1_hsi_220_label.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['hsi']
    hsi = d.Data # HWC
    lidar = d.Lidar[0].z
    truth = d.sceneLabels.labels
    truth[truth==-1] = 0
    truth = coo_array(truth, dtype='int')

    info = {
        'name': 'MUUFL Gulfport',
        'version': '0.1',
        'homepage': 'https://github.com/GatorSense/MUUFLGulfport',
        'license': 'MIT',
        'n_band_casi': hsi.shape[-1],
        'n_band_lidar': lidar.shape[-1],
        'n_class': d.sceneLabels.Materials_Type.size,
        'width': hsi.shape[1],
        'height': hsi.shape[0],
        'label_dict': dict(enumerate(d.sceneLabels.Materials_Type, start=1))
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), truth, info

def choice_coo_array(a, n_samples=20, n_class=11, seed=0x0d000721):
    np.random.seed(seed)
    train = coo_array(([],([],[])),a.shape, dtype='int')
    for cid in range(1,n_class+1):
        N = len(a.data[a.data==cid])
        indice = np.random.choice(N, n_samples, replace=False)
        row = a.row[a.data==cid][indice]
        col = a.col[a.data==cid][indice]
        val = np.ones(len(row)) * cid
        train += coo_array((val, (row, col)), shape=a.shape, dtype='int')
    test = (a - train)
    return train.tocoo(),test.tocoo()


class Muufl(Dataset):
    """
    A preprocessed houston2013 dataset
    """

    def preprocess(self):
        # 归一化
        self.HSI  = skimage.exposure.rescale_intensity(self.HSI, out_range='float32')  # 默认[0,1]
        self.LIDAR = skimage.exposure.rescale_intensity(self.LIDAR, out_range='float32')  # 默认[0,1]

        # 填充
        r = self.patch_radius
        pad_shape = ((0, 0), (r, r), (r, r))  # 只pad后两个维度
        self.hsi  = np.pad(self.HSI, pad_shape, 'symmetric')
        self.lidar = np.pad(self.LIDAR, pad_shape, 'symmetric')

        # 划分数据集
        train_truth, test_truth = choice_coo_array(self.UNDIVIDED_TRUTH, n_samples=20, n_class=self.INFO['n_class'])
        self.TRUTH = train_truth if self.TRAIN else test_truth

    def __init__(self, root: Path = None, train=True, download=True, patch_radius=3):
        self.lidar = None
        self.hsi = None
        self.patch_radius = patch_radius
        self.TRAIN = train
        self.HSI, self.LIDAR, self.UNDIVIDED_TRUTH, self.INFO = fetch_muufl(root, download)
        self.TRUTH = None
        self.preprocess()

    def __len__(self):
        return self.TRUTH.count_nonzero()

    def __getitem__(self, index):
        r = self.patch_radius
        R = r+1 # 因为Python切片是左闭右开,所以用R来表示右边界

        i = self.TRUTH.row[index] + r  # 补上pad导致的偏移
        j = self.TRUTH.col[index] + r
        cid = self.TRUTH.data[index].item()

        x_hsi   = self.hsi  [:, i-r:i+R, j-r:j+R]
        x_lidar = self.lidar[:, i-r:i+R, j-r:j+R]
        y = np.eye(self.INFO['n_class'])[cid-1]

        # 额外信息: 当前点的坐标
        extras = {}
        if not self.TRAIN:
            extras = {
                "x": i-r,
                "y": j-r,
                "index": index,
                "class": cid
            }

        return x_hsi, x_lidar, y, extras


__all__ = ['Muufl', 'fetch_muufl']
