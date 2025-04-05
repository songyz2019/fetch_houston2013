# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
from fetch_houston2013 import fetch_houston2013, fetch_muufl, split_spmatrix, fetch_trento
from fetch_houston2013.torch import Muufl, Houston2013, Trento
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

    def test_fetch_houston2013(self):
        casi, lidar, train_truth, test_truth, info = fetch_houston2013()
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

if __name__ == '__main__':
    unittest.main()