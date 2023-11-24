**English** | [中文(简体)](./README.zh-hans.md)

---

This project is still working in progress

---

# fetch_houston2013
Download and load Houston 2013 Dataset (2013 IEEE GRSS Data Fusion Contest) like a buildin function in scikit-learn.

- Automaticlly download all needed files
- Support caching
- Verify checksums
- Show copyright of the dataset
- numpy format
- PyTorch support
- Use sparse matrix to representing ground truth, less memory usage and easier iteration
- Faster loading than `.mat`
- Need more testing before GA

![screenshot](screenshot.png)

## Usage
1. Install scikit-image
```bash
pip install scikit-image==0.22.0
pip install scipy
```
2. copy [fetch_houston2013.py](fetch_houston2013/fetch_houston2013.py) to your project
3. import and run
```python
from fetch_houston2013 import fetch_houston2013
casi, lidar, train_y, test_y, num_class = fetch_houston2013()
```

### PyTorch
1. Install scikit-image
```bash
pip install scikit-image==0.22.0
```
2. copy [Houston2013.py](houston2013/Houston2013.py) to your project
3. import and run
```python
from Houston2013 import Houston2013
trainset = DataLoader(Houston2013(train=True, patch_size=7), batch_size=32, shuffle=True)
testset  = DataLoader(Houston2013(train=False, patch_size=7))
```
## Troubleshooting
Remove `~/scikit_learn_data` to clean cache and try again.  
We download dataset from official website and pastbin.com.  

## Benchmark

```python
In [4]: from util.houton2013 import fetch_houston2013
   ...: from time import time
   ...: start_time = time()
   ...: fetch_houston2013()
   ...: print("fetch_houston2013 %.4f ms"  % (1000*(time() - start_time)))

fetch_houston2013 521.4026 ms

In [5]: from scipy.io import loadmat
   ...: start_time = time()
   ...: loadmat(os.path.expanduser('~/dataset/Houston2013/HSI.mat'))['HSI']
   ...: loadmat(os.path.expanduser('~/dataset/Houston2013/DSM.mat'))['DSM']
   ...: loadmat(os.path.expanduser('~/dataset/Houston2013/TR.mat'))['TR_map']
   ...: loadmat(os.path.expanduser('~/dataset/Houston2013/TE.mat'))['TE_map']
   ...: print("loadmat Time %.4f ms" % (1000*(time() - start_time)))

loadmat Time 979.6755 ms
```

## TODO(Maybe)
Replace scikit-image with imageio
Publish to pypi or provide whl

## Credits
Houston2013 dataset: https://hyperspectral.ee.uh.edu/?page_id=459  
paperswithcode: https://paperswithcode.com/dataset/houston  
The 2013_IEEE_GRSS_DF_Contest_Samples_VA.txt in this repo is exported from original 2013_IEEE_GRSS_DF_Contest_Samples_VA.roi.

## License
Apache-2.0
