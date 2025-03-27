# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import os
from os.path import exists, expanduser, join
from pathlib import Path
import warnings
import hashlib
from zipfile import ZipFile
import urllib
import urllib.request
from io import StringIO
import logging

import numpy as np
from numpy.typing import NDArray
import skimage
from scipy.sparse import coo_array


def _get_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = os.environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def _read_roi(path :Path, shape) -> coo_array:
    """
    读取ENVI软件导出roi文件得到的txt文件,得到一个稀疏矩阵图像

    用起来像字典

    :param path: 文件路径
    :return: An coo_array image representing the ROI
    """
    warnings.simplefilter("ignore", category=UserWarning) # Supress loadtxt's warning when data is empty

    img = coo_array(shape, dtype='uint')
    buf = ""
    cid = 1
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line == os.linesep or line == "":
                data=np.loadtxt(StringIO(buf), usecols=(2, 1), comments=';', dtype='uint')
                buf = ""
                if data.size > 0:
                    rows,cols = data.T
                    vals = cid*np.ones_like(rows)
                    cid += 1
                    img += coo_array((vals,(rows,cols)), shape=shape)
                    img.data[img.data>cid] = 0  # 清除重复像素点
            else:
                buf += line

            if line == "":
                break

    warnings.resetwarnings()

    return img.tocoo()


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
        assert sha256(root/filename) == checksum, f"Incorrect SHA256 for {filename}. Expect {checksum}, Actual {sha256(root/filename)}. {extra_message}"


def fetch_houston2013(datahome=None, download_if_missing=True):
    """Load the Houston2013 data-set in scikit-learn style

    Download it if necessary.

    :return casi, lidar 高光谱图像(144x349x1905), 激光雷达图像(1x349x1905)
    :return train_truth,test_truth 训练集真值和测试集真值, a 349x1905 coo_array
    :return info 相关信息
    """
    logger = logging.getLogger("fetch_houston2013")

    def fetch_houston2013zip(path: Path, download_if_missing: bool = True) -> Path:
        """Make sure `path` is the zip file of Houston2013, or raise FileNotFoundError
        """
        if not exists(path):
            if download_if_missing:
                url = "https://hyperspectral.ee.uh.edu/2egf4tg8hial13gt/2013_DFTC.zip"
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
                urllib.request.install_opener(opener)
                logger.info(f"Downloading {url}")
                urllib.request.urlretrieve(url, path)
                urllib.request.install_opener(urllib.request.build_opener()) # Reset

            else:
                raise FileNotFoundError(f"{path} not found")

        _verify_files(path.parent, {path.name: "f4d619d5cbcb09d0301038f1b8fe83def6c2d484334b7b8127740a00ecf7e245"})
        return path

    # 1. 准备数据
    DATA_HOME = Path(_get_data_home(datahome))
    ZIP_PATH = DATA_HOME / 'Houston2013.zip'
    UNZIPED_PATH = DATA_HOME / 'Houston2013/'
    FILES_PATH = UNZIPED_PATH/'2013_DFTC'
    FILES_SHA256 = {
        "2013_IEEE_GRSS_DF_Contest_CASI.hdr": "869be3459978b535b873bca98b1cf05066c7acca9c160b486a86efd775005e8d",
        "2013_IEEE_GRSS_DF_Contest_CASI.tif": "1440f38594e8e82cc1944c084fc138ef55a70af54122828e999c4fb438574c14",
        "2013_IEEE_GRSS_DF_Contest_LiDAR.hdr": "053c083de1cb0d9ad51c56964b29669733ef2c7db05997d4f4e0779ab2f6aade",
        "2013_IEEE_GRSS_DF_Contest_LiDAR.tif": "9f4facce8876ee84642d9cb03536baf0389506de97ddc01b73366fe4521de981",
        "2013_IEEE_GRSS_DF_Contest_Samples_TR.roi": "feedf41f7064d8f80cf2d9bda72fcbcc98b64658d01e519ad0b90b1ca88f1375",
        "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt": "16c69cf216535d7b4df2045b05d49c50a078609aa6d011a5e23e54f4cd08abda",
        "2013_IEEE_GRSS_DF_Contest_Samples_VA.zip": "aac7015c7a986063002a86eb7f7cc57ed6f14f5eaf3e9ca29c0cb1e63fd7e0d5",
        "copyright.txt": "63d908383566b1ff6fd259aa202e31dab9a629808919d87d94970df7ad25180d",
        "2013_IEEE_GRSS_DF_Contest_Samples_VA.txt": "768bb02193d04c8020b45f1f31a49926a5b914040f77f71a81df756d6e8b8dcb"
    }
    if not exists(DATA_HOME):
        os.makedirs(DATA_HOME)

    # 2.下载数据集zip文件并解压
    if exists(FILES_PATH) and len(os.listdir(FILES_PATH)) > 0:  # 已存在解压的文件夹且非空
        _verify_files(FILES_PATH, FILES_SHA256, f"please try removing {FILES_PATH}")
    else:
        fetch_houston2013zip(ZIP_PATH, download_if_missing)
        # 解压 2013_DFTC 目录下的所有文件
        logger.info(f"Decompressing {ZIP_PATH}")
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)

        if not exists(FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt'):
            logger.info(f"Downloading 2013_IEEE_GRSS_DF_Contest_Samples_VA.txt")
            urllib.request.urlretrieve("https://pastebin.com/raw/FJyu5SQX", FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt')
            # Mirror: urllib.request.urlretrieve("https://github.com/songyz2019/fetch_houston2013/raw/main/data/2013_IEEE_GRSS_DF_Contest_Samples_VA.txt", FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt')
        _verify_files(FILES_PATH, FILES_SHA256)

        # 删除ZIP
        os.remove(ZIP_PATH)

        # 显示版权信息
        with open(FILES_PATH / 'copyright.txt', 'r', encoding='iso-8859-1') as f:
            logger.info(f.read())


    # 3. 数据加载
    lidar :NDArray = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_LiDAR.tif')[np.newaxis, :, :] # (1   349 1905)
    casi  :NDArray = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_CASI.tif' ).transpose(2,0,1)  # (144 349 1905)
    train_truth:coo_array= _read_roi(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_TR.txt', (349, 1905)) # (349 1905)
    test_truth :coo_array= _read_roi(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_VA.txt', (349, 1905)) # (349 1905)

    info = {
        'name': 'houston2013',
        'full_name': 'IEEE GRSS DF Contest Houston 2013',
        'homepage': 'https://hyperspectral.ee.uh.edu/?page_id=459',
        'n_band_hsi': 144,
        'n_band_lidar': 1,
        'n_class': 15,
        'width': 1905,
        'height': 349,
        'label_dict': {
            1 : 'Healthy grass',
            2 : 'Stressed grass',
            3 : 'Synthetic grass',
            4 : 'Trees',
            5 : 'Soil',
            6 : 'Water',
            7 : 'Residential',
            8 : 'Commercial',
            9 : 'Road',
            10: 'Highway',
            11: 'Railway',
            12: 'Parking Lot 1',
            13: 'Parking Lot 2',
            14: 'Tennis Court',
            15: 'Running Track'
        },
        "wavelength": np.array([
            364.000000,  368.799988,  373.600006,  378.399994,  383.200012,  387.899994,
            392.700012,  397.500000,  402.299988,  407.000000,  411.799988,  416.600006,
            421.399994,  426.100006,  430.899994,  435.700012,  440.500000,  445.200012,
            450.000000,  454.799988,  459.600006,  464.299988,  469.100006,  473.899994,
            478.600006,  483.399994,  488.200012,  492.899994,  497.700012,  502.500000,
            507.299988,  512.000000,  516.799988,  521.599976,  526.299988,  531.099976,
            535.900024,  540.599976,  545.400024,  550.200012,  554.900024,  559.700012,
            564.500000,  569.200012,  574.000000,  578.799988,  583.500000,  588.299988,
            593.099976,  597.799988,  602.599976,  607.400024,  612.099976,  616.900024,
            621.599976,  626.400024,  631.200012,  635.900024,  640.700012,  645.500000,
            650.200012,  655.000000,  659.799988,  664.500000,  669.299988,  674.099976,
            678.799988,  683.599976,  688.299988,  693.099976,  697.900024,  702.599976,
            707.400024,  712.200012,  716.900024,  721.700012,  726.500000,  731.200012,
            736.000000,  740.700012,  745.500000,  750.299988,  755.000000,  759.799988,
            764.599976,  769.299988,  774.099976,  778.900024,  783.599976,  788.400024,
            793.200012,  797.900024,  802.700012,  807.500000,  812.200012,  817.000000,
            821.799988,  826.500000,  831.299988,  836.099976,  840.799988,  845.599976,
            850.400024,  855.099976,  859.900024,  864.700012,  869.400024,  874.200012,
            879.000000,  883.700012,  888.500000,  893.299988,  898.000000,  902.799988,
            907.599976,  912.299988,  917.099976,  921.900024,  926.700012,  931.400024,
            936.200012,  941.000000,  945.799988,  950.500000,  955.299988,  960.099976,
            964.799988,  969.599976,  974.400024,  979.200012,  983.900024,  988.700012,
            993.500000,  998.299988,  1003.099976, 1007.799988, 1012.599976, 1017.400024,
            1022.200012, 1026.900024, 1031.699951, 1036.500000, 1041.300049, 1046.099976
        ])
    }

    return casi, lidar, train_truth, test_truth, info


__all__ = ['fetch_houston2013']


