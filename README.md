# fetch_houston2013

[![PyPI - Version](https://img.shields.io/pypi/v/fetch-houston2013.svg)](https://pypi.org/project/fetch-houston2013)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fetch-houston2013.svg)](https://pypi.org/project/fetch-houston2013)

Download and load Houston 2013 Dataset (2013 IEEE GRSS Data Fusion Contest) like a buildin function in scikit-learn.

- Automatically download and cache all needed files
- Verify checksums to avoid data poisoning
- Show copyright of the dataset
- Use sparse matrix to representing ground truth, less memory usage and easier iteration

![screenshot](screenshot.jpg)

## Usage
1. install this package
```bash
pip install fetch-houston2013
```
2. import and get the dataset
```python
from fetch_houston2013 import fetch_houston2013
hsi, dsm, train_label, test_label, info = fetch_houston2013()
```

## Troubleshooting
Remove `~/scikit_learn_data` to clean cache and try again.  
We download dataset from official website and pastbin.com. Make sure you can access these websites.

## Build and Publish
```bash
# install uv
uv build
bash load_secrets.bash
uv publish
```

## Credits
Houston2013 dataset: https://hyperspectral.ee.uh.edu/?page_id=459  
paperswithcode: https://paperswithcode.com/dataset/houston  
The 2013_IEEE_GRSS_DF_Contest_Samples_VA.txt in this repo is exported from original 2013_IEEE_GRSS_DF_Contest_Samples_VA.roi.

```text
Note: If this data is used in any publication or presentation the following reference must be cited:
P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, “MUUFL Gulfport Hyperspectral and LiDAR Airborne Data Set,” University of Florida, Gainesville, FL, Tech. Rep. REP-2013-570, Oct. 2013.
If the scene labels are used in any publication or presentation, the following reference must be cited:
X. Du and A. Zare, “Technical Report: Scene Label Ground Truth Map for MUUFL Gulfport Data Set,” University of Florida, Gainesville, FL, Tech. Rep. 20170417, Apr. 2017. Available: http://ufdc.ufl.edu/IR00009711/00001.
If any of this scoring or detection code is used in any publication or presentation, the following reference must be cited:
T. Glenn, A. Zare, P. Gader, D. Dranishnikov. (2016). Bullwinkle: Scoring Code for Sub-pixel Targets (Version 1.0) [Software]. Available from https://github.com/GatorSense/MUUFLGulfport/.
```

## License
Copyright 2025 songyz2023

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
