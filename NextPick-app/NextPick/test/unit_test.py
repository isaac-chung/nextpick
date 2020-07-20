import unittest
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import torch
import os

# print(os.getcwd())
# os.chdir('../')
# print(os.getcwd())

from NextPick.image_search import load_pretrained_model, transform, eval_test_image, create_df_for_map_plot
from NextPick.plotly_map import get_input_latlon, get_distances, get_top5_distance
from NextPick.ImageDataset import ImageDataset

# APP_PATH = '/home/ubuntu/application'
APP_PATH = '/mnt/c/Users/chung/Documents/04-Insight/nextpick/NextPick-app'
DATA_FOLDER = "%s/NextPick/data" %APP_PATH
BATCH = 100

# top_n images
TOP_N = 40
# annoy
ANNOY_PATH = '%s/NextPick/NextPick/annoy_idx_2.annoy' %APP_PATH
ANNOY_METRIC = 'angular'
RESNET18_FEAT = 512


class test_features(unittest.TestCase):
    '''
    Testing using the ski image.
    '''
    def setUp(self):
        self.model, self.model_full = load_pretrained_model()
        self.test_img = '%s/NextPick/static/assets/img/ski-test-img.png' % APP_PATH


    def test_feature(self):
        '''
        ski image feature embeddings
        '''
        feature = get_features(self.test_img, self.model)
        fname = '%s/NextPick/test/ski_test.pkl' %APP_PATH
        with open(fname, 'rb') as f:
            feature_loaded = pickle.load(f)
            f.close()
        np.testing.assert_array_almost_equal(feature[0], feature_loaded[0])


    def test_searches_index(self):
        '''
        Ski image search indices. The top-N closest results should be the same.
        '''
        # load the annoy index tree. eval_test_image function will fail if tree not loaded.
        if os.path.exists(ANNOY_PATH):
            annoy_idx_loaded = AnnoyIndex(RESNET18_FEAT, metric=ANNOY_METRIC)
            annoy_idx_loaded.load(ANNOY_PATH)
            print('Loaded annoy tree from memory.')
        searches = eval_test_image(self.test_img, self.model, annoy_idx_loaded, top_n=TOP_N)
        fname = '%s/NextPick/test/ski_searches.pkl' %APP_PATH
        with open(fname, 'rb') as f:
            searches_loaded = pickle.load(f)
            f.close()
        np.testing.assert_array_almost_equal(searches[0], searches_loaded[0])


    def test_searches_difference(self):
        '''
        Ski image search cosine differences.
        '''
        if os.path.exists(ANNOY_PATH):
            annoy_idx_loaded = AnnoyIndex(RESNET18_FEAT, metric=ANNOY_METRIC)
            annoy_idx_loaded.load(ANNOY_PATH)
            print('Loaded annoy tree from memory.')
        searches = eval_test_image(self.test_img, self.model, annoy_idx_loaded, top_n=TOP_N)
        fname = '%s/NextPick/test/ski_searches.pkl' %APP_PATH
        with open(fname, 'rb') as f:
            searches_loaded = pickle.load(f)
            f.close()
        np.testing.assert_array_almost_equal(searches[1], searches_loaded[1])


class test_df(unittest.TestCase):

    def setUp(self):
        self.model, self.model_full = load_pretrained_model()
        self.test_img = '%s/NextPick/static/assets/img/ski-test-img.png' % APP_PATH
        self.input_dataset = ImageDataset(DATA_FOLDER)
        self.pd_files = self.input_dataset.get_file_df()
        self.image_loader = torch.utils.data.DataLoader(self.input_dataset, batch_size=BATCH)
        fname = '%s/NextPick/test/ski_searches.pkl' % APP_PATH
        with open(fname, 'rb') as f:
            self.searches = pickle.load(f)
            f.close()


    def test_pd_files(self):
        '''
        Ski image pd_files
        '''
        fname_df = '%s/NextPick/test/ski_pd_files.pkl' % APP_PATH
        with open(fname_df, 'rb') as f:
            pd_files_loaded = pickle.load(f)
            f.close()
        self.assertTrue(pd_files_loaded.equals(self.pd_files))


    def test_create_df(self):
        '''
        Ski image df creation.
        '''
        df = create_df_for_map_plot(self.searches, self.pd_files) # df should have 40 rows
        fname = '%s/NextPick/test/ski_df_1.pkl' %APP_PATH
        with open(fname, 'rb') as f:
            df_loaded = pickle.load(f)
            f.close()
        self.assertTrue(df.equals(df_loaded))


def get_features(test_img, model):
    '''
    Search for the closest image as the test image.
    :param test_img: path of test image
    :param model:
    :param annoy_index:
    :return:
    '''
    with torch.no_grad():
        model.eval()
        data = Image.open(test_img, 'r').convert('RGB')
        data = transform(data).unsqueeze(0)
        feature = model(data)
    return feature


if __name__ == '__main__':
    unittest.main(verbosity=2)