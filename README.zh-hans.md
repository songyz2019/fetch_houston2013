[English](./README.md) | **中文(简体)**

# fetch_houston2013
下载并加载houston2013数据集

- 自动下载所有数据集(包括验证集)
- 缓存已下载的文件
- 验证下载文件是否正确
- numpy格式
- 用稀疏矩阵存储标签
- 支持PyTorch

![screenshot](screenshot.png)

## Usage
1. 安装 scikit-image
```bash
pip install scikit-image==0.22.0
```
2. 复制 [fetch_houston2013.py](fetch_houston2013/fetch_houston2013.py) 到你的项目中
3. 导入并运行
```python
from fetch_houston2013 import fetch_houston2013
casi, lidar, train_y, test_y, num_class = fetch_houston2013()
```

### PyTorch
1. 安装 scikit-image
```bash
pip install scikit-image==0.22.0
```
2. 复制 [Houston2013.py](houston2013/Houston2013.py) 到你的项目中
3. 导入并运行
```python
from Houston2013 import Houston2013
trainset = DataLoader(Houston2013(train=True, patch_size=7), batch_size=32, shuffle=True)
testset  = DataLoader(Houston2013(train=False, patch_size=7))
```
## 常见问题
删除 `~/scikit_learn_data` 来清理缓存
验证集从pastebin下载

## Credits
Houston2013 dataset: https://hyperspectral.ee.uh.edu/?page_id=459
paperswithcode: https://paperswithcode.com/dataset/houston

## License
反996许可证(Anti-996-License)
