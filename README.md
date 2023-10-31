# fetch_houston2013
Download and load Houston 2013 Dataset like a buildin function in scikit-learn.

- Automaticlly download all needed files
- Support caching
- Verify checksums
- Show copyright of the dataset
- numpy format

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
