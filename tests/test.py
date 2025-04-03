# SPDX-FileCopyrightText: 2025-present songyz2023 <songyz2023dlut@outlook.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
from fetch_houston2013 import fetch_houston2013, fetch_muufl, split_spmatrix
from fetch_houston2013.fetch_trento import fetch_trento


class Test(unittest.TestCase):
    def setUp(self):
        pass

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

    def test_torch_dataset(self):
        # 测试torch_dataset.py中的数据集
        from fetch_houston2013.torch import Houston2013, Trento, Muufl
        dataset = Houston2013(patch_radius=7, subset='train')
        x_h, x_l, y, extras = dataset[0]
        self.assertEqual(len(dataset), 2832)

        dataset = Muufl(patch_radius=11, subset='train')
        x_h, x_l, y, extras = dataset[0]
        self.assertEqual(len(dataset), 220)

        dataset = Trento(patch_radius=11, subset='train')
        x_h, x_l, y, extras = dataset[0]
        self.assertEqual(len(dataset), 120)

    def test_torch_muufl(self):
        from fetch_houston2013.torch import Muufl
        dataset = Muufl(subset='train')
        x_h, x_l, y, extras = dataset[0]
        self.assertEqual(x_h.shape, (64,11,11))
        self.assertEqual(x_l.shape, (1,11,11))
        self.assertEqual(y.shape, (dataset.n_class, ))


if __name__ == '__main__':
    unittest.main()