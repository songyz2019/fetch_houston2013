# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import skimage
from fetch_houston2013 import fetch_houston2013, fetch_muufl, split_spmatrix, fetch_trento
from fetch_houston2013.torch import Muufl, Houston2013, Trento
from fetch_houston2013.util import lbl2rgb
import torch
from torch.utils.data import DataLoader
from itertools import product


class Test(unittest.TestCase):
    def setUp(self):
        pass
    
    def torch_dataloader_test(self, dataset):
        b = 16
        dataset = Muufl(subset='train')
        dataloader = DataLoader(dataset, batch_size=b, shuffle=True)
        x_h, x_l, y, extras = next(iter(dataloader))
        
        n_test = 10
        for x_h, x_l, y, extras in dataloader:
            if torch.cuda.is_available():
                x_h, x_l, y = x_h.cuda(), x_l.cuda(), y.cuda()
            self.assertIsInstance(x_h, torch.Tensor)
            self.assertIsInstance(x_l, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(x_h.shape, torch.Size([b, 64,dataset.patch_size,dataset.patch_size]))
            self.assertEqual(x_l.shape, torch.Size([b, 1,dataset.patch_size,dataset.patch_size]))
            self.assertEqual(y.shape, torch.Size([b, dataset.n_class]))
            self.assertEqual(x_h.dtype, torch.float)
            self.assertEqual(x_l.dtype, torch.float)
            self.assertEqual(y.dtype, torch.float)
            if n_test <= 0:
                break
            else:
                n_test -= 1

    def generate_lbl2rgb(self, truth, info, subset):
        h,w = truth.shape
        y = np.eye(info['n_class']+1)[truth.todense()].transpose(2, 0, 1) # One Hot
        self.assertEqual(y.shape, (info['n_class']+1, h, w))

        rgb = lbl2rgb(y, info['name'])
        self.assertEqual(rgb.shape, (3, h, w))
        self.assertLessEqual(rgb.max(), 1.0)
        self.assertGreaterEqual(rgb.min(), 0.0)

        img = (rgb*255.0).astype(np.uint8).transpose(1, 2, 0)
        skimage.io.imsave(f"dist/{info['name']}_{subset}.png", img)

    def test_fetch_houston2013(self):
        casi, lidar, train_truth, test_truth, info = fetch_houston2013()

        self.assertEqual(train_truth.data.max(), info['n_class'])
        self.assertEqual(train_truth.data.min(), 1)
        self.assertEqual(train_truth.todense().min(), 0)
        self.assertEqual(test_truth.data.max(), info['n_class'])
        self.assertEqual(test_truth.data.min(), 1)
        self.assertEqual(test_truth.todense().min(), 0)

        self.assertEqual(casi.shape, (144, 349, 1905))
        self.assertEqual(lidar.shape, (1, 349, 1905))
        self.assertEqual(train_truth.shape, (349, 1905))
        self.assertEqual(test_truth.shape, (349, 1905))
        self.assertEqual(info['n_band_hsi'], 144)
        self.assertEqual(info['n_band_lidar'], 1)
        self.assertEqual(info['n_class'], 15)
        self.assertEqual(info['width'], 1905)
        self.assertEqual(info['height'], 349)
        self.assertEqual(len(info['label_dict']), 15)
        self.assertEqual(info['wavelength'].shape, (144,))
        self.assertEqual(info['wavelength'][0], 364.000000)
        self.assertEqual(info['wavelength'][-1], 1046.099976)

    def test_fetch_muufl(self):
        casi, lidar, truth, info = fetch_muufl()
        train_label, test_label = split_spmatrix(truth, 20)
        self.assertEqual(casi.shape, (64, 325, 220))
        self.assertEqual(lidar.shape, (2, 325, 220))
        self.assertEqual(truth.shape, (325, 220))
        self.assertEqual(train_label.shape, (325, 220))
        self.assertEqual(test_label.shape, (325, 220))

    def test_fetch_trento(self):
        casi, lidar, truth, info = fetch_trento()
        train_label, test_label = split_spmatrix(truth, 20)
        self.assertEqual(casi.shape, (63, 166, 600))
        self.assertEqual(lidar.shape, (2, 166, 600))
        self.assertEqual(truth.shape, (166, 600))
        self.assertEqual(train_label.shape, (166, 600))
        self.assertEqual(test_label.shape, (166, 600))

    def test_torch_datasets(self):
        for Dataset, subset, patch_size in product([Houston2013, Muufl, Trento], ['train', 'test', 'full'], [1, 5, 10, 11]):
            dataset = Dataset(subset=subset, patch_size=patch_size)
            self.torch_dataloader_test(dataset)

    def test_lbl2rgb(self):
        print(f"test_lbl2rgb needs visual verification.")
        for datafetch in [fetch_muufl, fetch_trento]:
            casi, lidar, truth, info = datafetch()
            train_label, test_label = split_spmatrix(truth, 100)
            self.generate_lbl2rgb(train_label, info, subset='train')
            self.generate_lbl2rgb(test_label, info, subset='test')
        for datafetch in [fetch_houston2013]:
            casi, lidar, train_label, test_label, info = datafetch()
            self.generate_lbl2rgb(train_label, info, subset='train')
            self.generate_lbl2rgb(test_label, info, subset='test')
        

if __name__ == '__main__':
    unittest.main()