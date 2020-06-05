# https://github.com/CSAILVision/places365
# https://github.com/surya501/reverse-image-search


from annoy import AnnoyIndex
from geopy.geocoders import Nominatim
import logging
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import pickle
from PIL import Image
import pandas as pd
import plotly.express as px
import seaborn as sns
sns.set_style("darkgrid")
from time import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
from torch.utils.data import Dataset, DataLoader


# define image transformer
transform = trn.Compose([trn.Resize((256,256)),
                               trn.CenterCrop(224),
                               trn.ToTensor(),
                               trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])

def load_pretrained_model(arch='resnet18'):
    '''
    Loads the pretrained PyTorch CNN models pretrained on places365 dataset.
    Model options are 'alexnet','densenet161','resnet18', and 'resnet50'.
    By default the 'resnet18' architecture is chosen for its lowest memory.
    Class labels follow 'categories_places365.txt'.
    :return: pretrained PyTorch model
    '''

    # make sure os.getcwd() returns the project home directory.
    model_file = 'places365/%s_places365.pth.tar' %arch

    # load pre-trained weights
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.fc_backup = model.fc
    model.fc = nn.Sequential()
    # model.eval()

    return model


def load_pkl_paths(folder):
    '''
    :param folder: the path of the 'data' folder
    :return: a list of paths to pickle files that store the geo data
    '''
    # pass in the 'data' folder as 'folder'
    class_names = [fold for fold in os.listdir(folder)]  # this should get folder names at the 'abbey' level
    paths_list = []
    for cl in class_names:
        img_files = [f for f in os.listdir(os.path.join(folder, cl)) if '.pkl' in f]

        for img in img_files:
            full_path = os.path.join(folder, cl, img)
            paths_list.append(full_path)

    df = pd.DataFrame(paths_list, columns=['path'])
    return df


def load_data_paths(folder):
    class_names = [fold for fold in os.listdir(folder)]  # this should get folder names at the 'abbey' level
    img_paths_list = []
    labels_list = []
    names_list = []

    for cl in class_names:
        # skip all .pkl files
        img_files = [f for f in os.listdir(os.path.join(folder, cl)) if '.pkl' not in f]
        for img in img_files:
            full_path = os.path.join(folder, cl, img)
            img_paths_list.append(full_path)
            labels_list.append(cl)
            names_list.append(img)

    df = pd.DataFrame(columns=['path', 'label', 'name'])
    df['path'] = img_paths_list
    df['label'] = labels_list
    df['name'] = names_list
    return df


class ImageDataset(Dataset):
    """
    A custom dataset to provide a batch of Images for hashing.
    Images are normalized for Imagenet mean and sd.
    Use the pandas dataframe get_file_df() for illustration purposes.
    """
    transform = None

    def __init__(self, path):
        self.df_files = load_data_paths(path)
        self.transform = trn.Compose([trn.Resize((256, 256)),
                               trn.CenterCrop(224),
                               trn.ToTensor(),
                               trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])

    def __getitem__(self, index):
        data = Image.open(self.df_files.iloc[index].path, 'r')
        data = data.convert('RGB')
        tensor = self.transform(data)
        return tensor, index

    def __len__(self):
        return len(self.df_files)

    def get_file_df(self):
        return self.df_files


def getVectorIndex(model, image_loader):
    """
    Evaluate model to get vector embeddings (features) and build tree for annoy indexing
    Arguments:
        model -- pre-trainned imagenet model to use.
        image_loader -- pytorch image_loader
    Returns:
        [annoy index] -- can be used for querying
    """
    t = AnnoyIndex(512, metric='angular')  # 512 for resnet18
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(image_loader):
            outputs = model(data)
            for i in range(0, len(data)):
                t.add_item(target[i], outputs[i])
    t.build(20)
    return t


def evalTestImage(test_img, model, annoy_index, top_n=5):
    '''
    Search for the closest image as the test iamge.
    :param test_img: path of test image
    :param model:
    :param annoy_index:
    :return:
    '''
    with torch.no_grad():
        model.eval()
        data = Image.open(test_img,'r').convert('RGB')
        data = transform(data).unsqueeze(0)
#         img_loader = DataLoader(data)
#         for idx, img in enumerate(img_loader):
        feature = model(data)
    searches = annoy_index.get_nns_by_vector(feature[0], top_n, include_distances=True)
    return searches[0], searches[1]


def plot_input_and_similar(test_img, searches, pd, titles=None):
    '''
    Use this function instead of plot_similar.
    test_img: text path to test image
    searches: tuple of lists. index[0] is a list of similar indices. index[1] is a list of cosine distances (flat_list)
    pd: dataframe of paths
    '''
    idx = searches[0]
    titles = searches[1]

    f = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(left=0.1, bottom=0, right=0.2, top=1.02, wspace=0.02, hspace=0.02)
    plt.axis('Off')
    rows = len(idx)
    cols = 2
    for i, img_idx in enumerate(idx):
        sp = f.add_subplot(rows, cols, 2 * (i + 1))  # want the output pictures on the right side
        sp.axis('Off')
        sp.get_xaxis().set_visible(False)
        sp.get_yaxis().set_visible(False)
        if titles is not None:
            sp.set_title('Cosine sim = %.3f' % titles[i], fontsize=16)
        data = Image.open(pd.iloc[img_idx].path, 'r')
        data = data.convert('RGB')
        data = data.resize((400, 300), Image.ANTIALIAS)
        plt.imshow(data)
        plt.tight_layout()

    # plot test image
    sp = f.add_subplot(rows, cols, rows)  # want the test image in the middle of the column
    sp.axis('Off')
    sp.get_xaxis().set_visible(False)
    sp.get_yaxis().set_visible(False)
    sp.set_title('User Input', fontsize=16)
    data = Image.open(test_img, 'r')
    data = data.convert('RGB')
    data = data.resize((400, 300), Image.ANTIALIAS)
    plt.imshow(data)
    plt.tight_layout()
    plt.autoscale(tight=True)


def create_df_for_map_plot(searches, pd_files):
    '''
    :param searches: search results from evalTestImage()
    :param pd_files: pandas DataFrame from ImageDataset.get_file_df()
    :return: DataFrame of the searches' location and address
    '''
    idx = searches[0]  # list of annoy index results
    class_labels = list(pd_files.iloc[idx].label.drop_duplicates())
    name = list(pd_files.iloc[idx]['name'].str.rstrip('.jpg'))  # list of photo id's

    for_plotly = pd.DataFrame(columns=['latitude', 'longitude'])
    for label in class_labels:
        with open('notebooks/data/%s/%s.pkl' % (label, label), 'rb') as f:
            locations = pickle.load(f)
            for_plotly = pd.concat(
                [locations.loc[locations['id'].isin(name)][['latitude', 'longitude']], for_plotly])
            f.close()

    for_plotly = for_plotly.reset_index(drop=True)
    keys_for_display = ['country']
    gl = Nominatim(user_agent='default')
    for_plotly['latlon'] = list(zip(for_plotly['latitude'], for_plotly['longitude']))
    locations = []
    display_names = []

    for i, row in for_plotly.iterrows():
        location = gl.reverse(for_plotly.iloc[i]['latlon'])
        locations.append(location.address)
        display_names.append(", ".join([location.raw['address'][key] for key in keys_for_display]))
    for_plotly['address'] = locations
    for_plotly['display'] = display_names

    return for_plotly


def plot_map(searches, pd_files):
    '''
    Plots search results on a map
    :param searches: search results from evalTestImage()
    :param pd_files: pandas DataFrame from ImageDataset.get_file_df()
    '''
    df = create_df_for_map_plot(searches, pd_files)
    fig = px.scatter_geo(df, lat='latitude', lon='longitude', text='display')
    fig.show()


if __name__ == '__main__':
    '''
    load annoy index and make predictions / recommendations
    '''

    input_dataset = ImageDataset('notebooks/data')
    bs = 100
    image_loader = torch.utils.data.DataLoader(input_dataset, batch_size=bs)
    model = load_pretrained_model()

    pd_files = input_dataset.get_file_df()

    annoy_path = 'notebooks/annoy_idx.annoy'

    if os.path.exists(annoy_path):
        annoy_idx_loaded = AnnoyIndex(512)
        annoy_idx_loaded.load(annoy_path)

    test_img = 'notebooks/ski-test-img.png'
    searches = evalTestImage(test_img, model, annoy_idx_loaded)

    # plot images and map. Note that these plor functions call
    # create_df_for_map_plot, and also finds the path of the images
    # from the indices (not in that order). Would be useful for web app.
    plot_input_and_similar(test_img, searches, pd_files)
    plot_map(searches, pd_files)