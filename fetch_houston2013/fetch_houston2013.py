import hashlib
import os
import urllib
from io import StringIO
from os import makedirs
from os.path import exists, expanduser, join
from typing import List
from urllib.request import urlretrieve
import numpy as np
from zipfile import ZipFile
from pathlib import Path
from numpy.typing import NDArray
import skimage

def index2onehot(x :NDArray, num_class):
    return np.eye(num_class)[x]


def get_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = os.environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
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


def read_roi_txt(path :Path, shape) -> List[NDArray]: # TODO: 性能问题
    result = []
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

    m = np.zeros(shape, dtype='uint8')
    for id,indices in enumerate(result):
        for i in indices:
            m[*i] = id+1
    return m


def verify_files(root: Path, files_sha256: dict) -> bool:
    """验证root下的文件的sha256是否与files_sha256相符

    :param files_sha256: 例如: `{"1.txt", "f4d619....", "2.txt": "9d03010....."}`
    :param root: 文件夹目录
    """
    return all([sha256(root / filename) == checksum for filename, checksum in files_sha256.items()])


def fetch_houston2013(datahome=None, download_if_missing=True):
    """Load the Houston2013 data-set (classification).

    Download it if necessary. casi is a 144 channel image (144 349 1905), lidar is an 1 channel image(1 349 1905), train_y and test_y is the onehot ground truth (349 1905 16) with 16 classes.

    :return casi, lidar, train_y, test_y, num_class
    """

    def fetch_houston2013zip(path: Path, download_if_missing: bool = True) -> Path:
        """Make sure `path` is the zip file of Houston2013, or raise FileNotFoundError

        :return (casi:(349,1905,144), lidar:(349,1905))
        """
        if not exists(path):
            if download_if_missing:
                url = "https://hyperspectral.ee.uh.edu/2egf4tg8hial13gt/2013_DFTC.zip"
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
                urllib.request.install_opener(opener)
                print(f"Downloading {url}")
                urlretrieve(url, path)
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
        "2013_IEEE_GRSS_DF_Contest_CASI.hdr"      : "869be3459978b535b873bca98b1cf05066c7acca9c160b486a86efd775005e8d",
        "2013_IEEE_GRSS_DF_Contest_CASI.tif"      : "1440f38594e8e82cc1944c084fc138ef55a70af54122828e999c4fb438574c14",
        "2013_IEEE_GRSS_DF_Contest_LiDAR.hdr"     : "053c083de1cb0d9ad51c56964b29669733ef2c7db05997d4f4e0779ab2f6aade",
        "2013_IEEE_GRSS_DF_Contest_LiDAR.tif"     : "9f4facce8876ee84642d9cb03536baf0389506de97ddc01b73366fe4521de981",
        "2013_IEEE_GRSS_DF_Contest_Samples_TR.roi": "feedf41f7064d8f80cf2d9bda72fcbcc98b64658d01e519ad0b90b1ca88f1375",
        "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt": "16c69cf216535d7b4df2045b05d49c50a078609aa6d011a5e23e54f4cd08abda",
        "2013_IEEE_GRSS_DF_Contest_Samples_VA.zip": "aac7015c7a986063002a86eb7f7cc57ed6f14f5eaf3e9ca29c0cb1e63fd7e0d5",
        "2013_IEEE_GRSS_DF_Contest_Samples_VA.txt": "768bb02193d04c8020b45f1f31a49926a5b914040f77f71a81df756d6e8b8dcb",
        "copyright.txt"                           : "63d908383566b1ff6fd259aa202e31dab9a629808919d87d94970df7ad25180d"
    }
    if not exists(DATA_HOME):
        makedirs(DATA_HOME)

    # 2.Download ZIP file and uncompress
    if exists(FILES_PATH) and len(os.listdir(FILES_PATH)) > 0:  # Already exists and not empty
        assert verify_files(FILES_PATH,FILES_SHA256), f"Dataset folder {FILES_PATH} already exists,but failed to verify!请考虑删除文件或更改data_home参数"
    else:
        fetch_houston2013zip(ZIP_PATH, download_if_missing)
        # extract all files in 2013_DFTC/
        with ZipFile(ZIP_PATH, 'r') as zip_file:
            zip_file.extractall(UNZIPED_PATH)
        
        if not exists(FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt'):
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36')]
            urllib.request.install_opener(opener)
            print(f"Downloading 2013_IEEE_GRSS_DF_Contest_Samples_VA.txt...")
            urlretrieve("https://pastebin.com/raw/FJyu5SQX", FILES_PATH/'2013_IEEE_GRSS_DF_Contest_Samples_VA.txt')

        assert verify_files(FILES_PATH, FILES_SHA256), f"Failed to verify unzipped files! (Maybe `2013_IEEE_GRSS_DF_Contest_Samples_VA.txt` download failed)"
        with open(FILES_PATH / 'copyright.txt', 'r', encoding='iso-8859-1') as f:
            print(f.read())

    # 3. 数据加载
    lidar  :NDArray = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_LiDAR.tif')[np.newaxis, :, :] # (1   349 1905)
    casi   :NDArray = skimage.io.imread(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_CASI.tif' ).transpose(2,0,1)  # (144 349 1905)
    train_y:NDArray = read_roi_txt(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_TR.txt' , lidar.shape[1:]) # (349 1905)
    test_y :NDArray = read_roi_txt(FILES_PATH / '2013_IEEE_GRSS_DF_Contest_Samples_VA.txt' , lidar.shape[1:]) # (349 1905)
    num_class = 16

    return casi, lidar, train_y, test_y, num_class


if __name__=='__main__':
    casi, lidar, train_y, test_y, num_class = fetch_houston2013()
    print(casi.shape)
    print(lidar.shape)
    print(train_y.shape)
    print(test_y.shape)
    