import os
from os.path import exists, expanduser, join
from pathlib import Path
import hashlib
from zipfile import ZipFile
import urllib
import urllib.request
import logging

import numpy as np
import scipy.io

from scipy.sparse import coo_array
from jaxtyping import Float


def fetch_trento(datahome=None, download_if_missing=True):
    """
    Donwload and load the Trento dataset.

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

        _verify_files(path.parent, {path.name: "B203331B039D994015C4137753F15973CB638046532B8DCED6064888BF970631".lower()})
        return path

    # 1. 准备
    logger = logging.getLogger("fetch_trento")
    URL = "https://github.com/tyust-dayu/Trento/archive/b4afc449ce5d6936ddc04fe267d86f9f35536afd.zip"
    DATA_HOME = Path(_get_data_home(datahome))
    ZIP_PATH = DATA_HOME / 'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd.zip'
    UNZIPED_PATH = DATA_HOME / 'MUUFLGulfport/'
    ROOT = UNZIPED_PATH/'Trento-b4afc449ce5d6936ddc04fe267d86f9f35536afd'
    FILES_SHA256 = {
        "allgrd.mat": "7e3fb2a2ea22c2661dfc768db3cb93c9643b324e7e64fadedfa57f5edbf1818f",
        "Italy_hsi.mat": "7b965fd405314b5c91451042e547a1923be6f5a38c6da83969032cff79729280",
        "Italy_lidar.mat": "a04dc90368d6a7b4f9d3936024ba9fef4105456c090daa14fff31b8b79e94ab1"
    }
    if not exists(DATA_HOME):
        os.makedirs(DATA_HOME)

    # 2.下载数据集zip文件并解压
    if exists(ROOT) and len(os.listdir(ROOT)) > 0:  # 已存在解压的文件夹且非空
        _verify_files(ROOT, FILES_SHA256, f"please try removing {ROOT}")
    else:
        fetch_zip(URL, ZIP_PATH, download_if_missing)
        logger.info(f"Decompressing {ZIP_PATH}")
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)

        _verify_files(ROOT, FILES_SHA256)

        # 删除ZIP
        os.remove(ZIP_PATH)

    # 3. 数据加载
    hsi = scipy.io.loadmat(
        ROOT / 'Italy_hsi.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    lidar = scipy.io.loadmat(
        ROOT / 'Italy_lidar.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['data']

    truth = scipy.io.loadmat(
        ROOT / 'allgrd.mat',
        squeeze_me=True,
        mat_dtype=True,
        struct_as_record=False
    )['mask_test']
    truth = coo_array(truth)

    info = {
        'name': 'MUUFL Gulfport',
        'version': '0.1',
        'homepage': 'https://github.com/GatorSense/MUUFLGulfport',
        'license': 'MIT',
        'n_band_casi': hsi.shape[-1],
        'n_band_lidar': lidar.shape[-1],
        # 'n_class': d.sceneLabels.Materials_Type.size,
        'width': hsi.shape[1],
        'height': hsi.shape[0],
        # 'label_dict': dict(enumerate(d.sceneLabels.Materials_Type, start=1))
    }

    return hsi.transpose(2,0,1), lidar.transpose(2,0,1), truth, info



__all__ = ['fetch_trento']
