def preprocess_houston2013(casi :NDArray, lidar :NDArray, train_y :NDArray, test_y :NDArray, patch_size=7, num_class=16):
    pad_width = (patch_size-1)//2
    casi_pixels = skimage.util.view_as_windows(casi,  window_shape=(144, 1, 1)).squeeze(0)                    # (349, 1905, 144, 1, 1)

    # 1 填充
    casi   = np.pad(casi, ((0,0),(pad_width,pad_width),(pad_width,pad_width)), 'symmetric') # 只pad后两个维度
    lidar  = np.pad(lidar, ((0,0),(pad_width,pad_width),(pad_width,pad_width)), 'symmetric')

    # 2 切片
    lidar       = skimage.util.view_as_windows(lidar, window_shape=(1,patch_size, patch_size)).squeeze(0)     # (349, 1905, 1,   7, 7)
    casi        = skimage.util.view_as_windows(casi,  window_shape=(144, patch_size, patch_size)).squeeze(0)  # (349, 1905, 144, 7, 7)

    # 3 滤出训练集像素, [性能瓶颈]
    indice = (train_y == 0)
    train_casi  = casi[indice]              # (662200 144 7 7)
    train_lidar = lidar[indice]             # (662200 1   7 7)
    train_casi_pixels = casi_pixels[indice] # (662200 144 1 1)
    train_y = train_y[indice]                   # (662200 16)

    # 3 滤出测试集像素, [性能瓶颈]
    indice = (test_y == 0)
    test_casi  = casi[indice]              # (662200 144 7 7)
    test_lidar = lidar[indice]             # (662200 1   7 7)
    test_casi_pixels = casi_pixels[indice] # (662200 144 1 1)
    test_train_y = train_y[indice]         # (662200 16)
    print(test_casi.shape)

    return (
        (train_casi,train_lidar,  train_casi_pixels, index2onehot(train_y, num_class)),
        (test_casi, test_lidar,    test_casi_pixels, index2onehot(test_y,  num_class))
    )




class Houston2013(Dataset):
    def __init__(self, root :Path=None, train=True, download=True, patch_size=7):
        self.train = train
        casi, lidar, train_y, test_y, self.num_class = fetch_houston2013(root, download, minmax_scale=True)
        (
            (self.train_casi, self.train_lidar, self.train_casi_pixels, self.train_y),
            (self.test_casi,  self.test_lidar,  self.test_casi_pixels,  self.test_y),
        ) = preprocess_houston2013(casi, lidar, train_y, test_y, patch_size=patch_size)

    def __len__(self):
        if self.train:
            return len(self.train_casi)
        else:
            return len(self.test_casi)

    def __getitem__(self, index):
        casi, lidar, casi_pixels, y = None, None, None, None
        if self.train:
            casi, lidar, casi_pixels, y = self.train_casi, self.train_lidar, self.train_casi_pixels, self.train_y
        else:
            casi, lidar, casi_pixels, y = self.test_casi, self.test_lidar, self.test_casi_pixels, self.test_y
        x_casi = casi[index]
        x_casi_pixels= casi_pixels[index]
        x_lidar= lidar[index]
        y = y[index]
        return x_casi,x_lidar,x_casi_pixels,y

