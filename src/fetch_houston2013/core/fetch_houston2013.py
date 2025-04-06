# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import os
from os.path import exists
from pathlib import Path
from typing import Optional
from zipfile import ZipFile
import urllib
import urllib.request
import logging

import numpy as np
import rasterio
from scipy.sparse import coo_array, spmatrix
from jaxtyping import UInt16, Float32, UInt64

from ..util.fileio import get_data_home, verify_files, read_roi
from .common import DataMetaInfo

def fetch_houston2013(datahome: Optional[str] = None, download_if_missing=True) -> tuple[
    UInt16[np.ndarray, '144 349 1905'],
    Float32[np.ndarray, '1 349 1905'],
    UInt64[spmatrix, '349 1905'],
    UInt64[spmatrix, '349 1905'],
    DataMetaInfo
]:
    """Load the Houston2013 data-set in scikit-learn style

    Download it if necessary. All the image are CHW formats. And the shape are typed in the return type.
    The background label is 0, and the rest are 1-15.
    
    :param datahome: The path to store the data files, default is '~/scikit_learn_data'
    :param download_if_missing: Whether to download the data if it is not found

    :return: (hsi, dsm, train_truth, test_truth, info)
    """
    logger = logging.getLogger("fetch_houston2013")

    def fetch_houston2013zip(path: Path, download_if_missing: bool = True) -> Path:
        """Make sure `path` is the zip file of Houston2013, or raise FileNotFoundError
        """
        if not exists(path):
            if download_if_missing:
                url = "https://machinelearning.ee.uh.edu/2egf4tg8hial13gt/2013_DFTC.zip"
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
                urllib.request.install_opener(opener)
                logger.info(f"Downloading {url}")
                urllib.request.urlretrieve(url, path)
                urllib.request.install_opener(urllib.request.build_opener()) # Reset

            else:
                raise FileNotFoundError(f"{path} not found")

        verify_files(path.parent, {path.name: "f4d619d5cbcb09d0301038f1b8fe83def6c2d484334b7b8127740a00ecf7e245"})
        return path

    # 1. 准备数据
    DATA_HOME = Path(get_data_home(datahome))
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
        verify_files(FILES_PATH, FILES_SHA256, f"please try removing {FILES_PATH}")
    else:
        fetch_houston2013zip(ZIP_PATH, download_if_missing)
        # 解压 2013_DFTC 目录下的所有文件
        logger.info(f"Decompressing {ZIP_PATH}")
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)

        if not exists(FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt'):
            logger.info(f"Downloading 2013_IEEE_GRSS_DF_Contest_Samples_VA.txt")
            va_urls = ["https://github.com/songyz2019/fetch_houston2013/raw/8539c932284a0d2ae60e7f968f430c42d4d1c09a/data/2013_IEEE_GRSS_DF_Contest_Samples_VA.txt", 'https://pastebin.com/raw/FJyu5SQX']
            for va_url in va_urls:
                try:
                    urllib.request.urlretrieve(va_url, FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt')
                except Exception as e:
                    print(f"Error fetching {va_url}, using mirrors.")
                else:
                    break
        verify_files(FILES_PATH, FILES_SHA256)

        # 删除ZIP
        os.remove(ZIP_PATH)

        # 显示版权信息
        with open(FILES_PATH / 'copyright.txt', 'r', encoding='iso-8859-1') as f:
            logger.info(f.read())


    # 3. 数据加载
    with rasterio.open(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_LiDAR.tif') as f:
        lidar = f.read()
    with rasterio.open(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_CASI.tif') as f:
        casi = f.read()

    train_truth:coo_array= read_roi(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_TR.txt', (349, 1905)) # (349 1905)
    test_truth :coo_array= read_roi(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_VA.txt', (349, 1905)) # (349 1905)

    info :DataMetaInfo = {
        'name': 'houston2013',
        'full_name': 'IEEE GRSS DF Contest Houston 2013',
        'homepage': 'https://machinelearning.ee.uh.edu/?page_id=459',
        'n_channel_hsi': 144,
        'n_channel_lidar': 1,
        'n_class': 15,
        'width': 1905,
        'height': 349,
        'label_name': {
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


