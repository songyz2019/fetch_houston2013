import hashlib
import os
from os.path import exists, expanduser, join
from zipfile import ZipFile
from pathlib import Path
import urllib
import urllib.request
import warnings
from io import StringIO
from typing import List

import numpy as np
from numpy.typing import NDArray

import skimage

from torch.utils.data import Dataset


def get_nth_in_sublist(l: List[List], n: int):
    """返回一个List[List[]]中绝对的第N个元素

    例如: get_nth_in_sublist([[],[1,2,3],[4,5]], 4) 会 返回 3
    """
    j = n
    length_list = enumerate([len(c) for c in l])
    for i, length in length_list:
        if j < length:
            return l[i][j], i
        else:
            j -= length
    raise IndexError(f"index out of range {n}/{sum(length_list)}")


def get_data_home(data_home=None) -> str:
    """Return the path of the scikit-learn data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default, the data directory is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/sklearn_learn_data`.

    Returns
    -------
    data_home: str
        The path to scikit-learn data directory.
    """
    if data_home is None:
        data_home = os.environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
    data_home = expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


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


def read_roi_txt_as_list(path :Path) -> List[NDArray]:
    """
    读取ENVI软件导出roi文件得到的txt文件,得到一个 *List*

    用起来像字典

    :param path: 文件路径
    :return: 示例 [NotImplemented, [(1,2),(3,4),(5,6)], [(7,8)], [(9,10)] ], 表示第一类(比如健康草地)有如下像素点(1,2),(3,4),(5,6), 第二类(干草地)有如下像素点(7,8)。第0个元素是地面,不应该使用它
    """
    warnings.simplefilter("ignore", category=UserWarning) # Supress loadtxt's warning when data is empty

    result = [[],] # 不应该访问第0个元素,它是地面
    buf = ""
    with open(path, 'r') as f:
        for line in f.readlines():
            if line == os.linesep:
                data=np.loadtxt(StringIO(buf), usecols=(2, 1), comments=';', dtype='uint')
                buf = ""
                if data.size > 0:
                    result.append(data)
            else:
                buf += line

    warnings.resetwarnings()
    return result


def verify_files(root: Path, files_sha256: dict) -> bool:
    """验证root下的文件的sha256是否与files_sha256相符

    :param files_sha256: 例如: `{"1.txt", "f4d619....", "2.txt": "9d03010....."}`
    :param root: 文件夹目录
    """
    return all([sha256(root / filename) == checksum for filename, checksum in files_sha256.items()])


def fetch_houston2013(datahome=None, download_if_missing=True):
    """Load the Houston2013 data-set in scikit-learn style

    Download it if necessary.

    :return casi, lidar, train_y, test_y 高光谱图像(144x349x1905), 激光雷达图像(1x349x1905)
    :return train_truth,test_truth 训练集真值和测试集真值, train_truth[i][j] 代表第i类的第j个像素坐标
    :return label_dict 编号->标签名
    """

    def fetch_houston2013zip(path: Path, download_if_missing: bool = True) -> Path:
        """Make sure `path` is the zip file of Houston2013, or raise FileNotFoundError
        """
        if not exists(path):
            if download_if_missing:
                url = "https://hyperspectral.ee.uh.edu/2egf4tg8hial13gt/2013_DFTC.zip"
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
                urllib.request.install_opener(opener)
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, path)
            else:
                raise FileNotFoundError(f"{path} not found")
        else:
            print(f"{path} already exists")
        assert sha256(path) == "f4d619d5cbcb09d0301038f1b8fe83def6c2d484334b7b8127740a00ecf7e245"
        return path

    # 1. 准备数据
    DATA_HOME = Path(get_data_home(datahome))
    ZIP_PATH = DATA_HOME / 'Houston2013.zip'
    UNZIPED_PATH = DATA_HOME / 'Houston2013/' # 解压到
    FILES_PATH = UNZIPED_PATH/'2013_DFTC' # 文件根目录
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
        assert verify_files(FILES_PATH,FILES_SHA256), f"数据集文件夹{FILES_PATH}已存在,但未通过哈希验证!请考虑删除文件或更改data_home参数"
    else:
        fetch_houston2013zip(ZIP_PATH, download_if_missing)
        # 解压 2013_DFTC 目录下的所有文件
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)
        
        if not exists(FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt'):
            urllib.request.urlretrieve("https://pastebin.com/raw/FJyu5SQX", FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt')

        assert verify_files(FILES_PATH, FILES_SHA256), f"解压后的数据未通过哈希验证(可能是2013_IEEE_GRSS_DF_Contest_Samples_VA.txt下载失败)"
        with open(FILES_PATH / 'copyright.txt', 'r', encoding='UTF-8') as f:
            print(f.read())

    # 3. 数据加载
    lidar :NDArray = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_LiDAR.tif')[np.newaxis, :, :] # (1   349 1905)
    casi  :NDArray = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_CASI.tif' ).transpose(2,0,1)  # (144 349 1905)
    train_truth:List= read_roi_txt_as_list(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_TR.txt') # A List of locations for each class, for example: [ [(1,2),(3,4),(5,6)], [(7,8)], [(9,10)] ]
    test_truth :List= read_roi_txt_as_list(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_VA.txt') # A List of locations for each class
    label_dict = {
        0 : 'Ground',
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
    }

    return casi, lidar, train_truth, test_truth, label_dict


class Houston2013(Dataset):
    """
    A preprocessed houston2013 dataset
    """
    def preprocess(self):
        # 归一化
        self.casi = skimage.exposure.rescale_intensity( self.casi, out_range='float32')  # 默认[0,1]
        self.lidar = skimage.exposure.rescale_intensity(self.lidar, out_range='float32')  # 默认[0,1]

        # 获得casi_pixels
        self.casi_pixels = skimage.util.view_as_windows(self.casi, window_shape=(144, 1, 1)).squeeze(0)  # (349, 1905, 144, 1, 1)

        # 填充
        pad_width = (self.patch_size - 1) // 2
        self.casi =  np.pad(self.casi,  ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), 'symmetric')  # 只pad后两个维度
        self.lidar = np.pad(self.lidar, ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), 'symmetric')

        # 切片
        self.lidar = skimage.util.view_as_windows(self.lidar, window_shape=(1,   self.patch_size, self.patch_size)).squeeze(0)  # (349, 1905, 1,   7, 7)
        self.casi =  skimage.util.view_as_windows(self.casi,  window_shape=(144, self.patch_size, self.patch_size)).squeeze(0)  # (349, 1905, 144, 7, 7)

    def __init__(self, root :Path = None, train=True, download=True, patch_size=7, exclude_ground=True):
        self.patch_size = patch_size
        self.train = train
        self.exclude_ground = exclude_ground

        self.casi, self.lidar, self.train_truth, self.test_truth, self.label_dict = fetch_houston2013(root, download)

        self.truth = self.train_truth if self.train else self.test_truth
        self.num_class = len(self.label_dict) - self.exclude_ground

        self.preprocess()

    def __len__(self):
        return sum([ sum([len(c) for c in self.truth]) ])

    def __getitem__(self, index):
        (i,j),claz = get_nth_in_sublist(self.truth, index)
        y = np.eye(self.num_class)[claz-self.exclude_ground] # OneHot
        return self.casi[i,j], self.lidar[i,j], self.casi_pixels[i,j], y


__all__ = ['Houston2013','fetch_houston2013']
