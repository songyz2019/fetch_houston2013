[English](./README.md) | [中文(简体)](./README.zh-hans.md)

# fetch_houston2013
Download and load Houston 2013 Dataset like a buildin function in scikit-learn.

- Automaticlly download all needed files
- Support caching
- Verify checksums
- Show copyright of the dataset
- numpy format
- PyTorch support

![screenshot](screenshot.png)

## Usage
1. Install scikit-image
```bash
pip install scikit-image==0.22.0
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

## TODO(Maybe)
Replace scikit-image with imageio

## Credits
Houston2013 dataset: https://hyperspectral.ee.uh.edu/?page_id=459
paperswithcode: https://paperswithcode.com/dataset/houston

## License
Apache-2.0
