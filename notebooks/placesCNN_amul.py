# https://github.com/CSAILVision/places365
# https://github.com/surya501/reverse-image-search

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
from torch.utils.data import Dataset, DataLoader
from annoy import AnnoyIndex
import os
import pickle
from PIL import Image
import logging
import pandas as pd

# define image transformer
center_crop = trn.Compose([trn.Resize((256,256)),
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


def load_classes(file_name='places365/categories_places365.txt'):
    '''
    :param file_name:
    :return: tuple of classes
    '''
    # make sure os.getcwd() returns the project home directory.
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return classes


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
        test_input = ImageDataset(test_img)
        img_loader = DataLoader(test_input)
        for _, (data, target) in enumerate(img_loader):
            feature = model(data)
    searches = annoy_index.get_nns_by_vector(feature, top_n, include_distances=True)
    return searches


if __name__ == '__main__':
    import os
    print(os.getcwd())  # C:\Users\chung\Documents\04-Insight\insight\notebooks
    os.chdir('../')
    print(os.getcwd())  # C:\Users\chung\Documents\04-Insight\insight

    input_dataset = ImageDataset('notebooks/data')
    bs = 100
    image_loader = torch.utils.data.DataLoader(input_dataset, batch_size=bs)
    model = load_pretrained_model()

    # TO DO: store these indices.
    # TO DO: look into ensemble methods? (week 3)
    annoy_idx = getVectorIndex(model, image_loader)
