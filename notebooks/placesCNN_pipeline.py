# https://github.com/CSAILVision/places365
# https://github.com/hundredblocks/semantic-search
# https://github.com/surya501/reverse-image-search

import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from torch.utils.data import Dataset
from annoy import AnnoyIndex
import os
import pickle
from PIL import Image
from time import time
import logging

# define image transformer
center_crop = trn.Compose([trn.Resize((256,256)),
                               trn.CenterCrop(224),
                               trn.ToTensor(),
                               trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])

gray_center_crop = trn.Compose([trn.Resize((256)),
                                trn.CenterCrop(224),
                                trn.ToTensor(),
                                trn.Normalize((0.5), (0.5))

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
    model.eval()

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


def load_img_paths(folder):
    '''
    :param folder: the path of the 'data' folder
    :return: a list of paths to images
    '''
    # pass in the 'data' folder as 'folder'
    class_names = [fold for fold in os.listdir(folder)] # this should get folder names at the 'abbey' level
    paths_list = []
    for cl in class_names:
        # skip all .pkl files
        img_files = [f for f in os.listdir(os.path.join(folder, cl)) if '.pkl' not in f]

        for img in img_files:
            full_path = os.path.join(folder, cl, img)
            paths_list.append(full_path)

    return paths_list


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

    return paths_list


def generate_features(arch, img_paths):
    '''
    :param img_name: example: 'notebooks/data/forest path/49159594286.jpg'
    :return: top 5 probabilities and class labels
    '''
    logging.info('Generating features...')
    start = time()
    model = load_pretrained_model(arch)
    # imgs = torch.zeros(len(img_paths), 3, 224, 224)
    # imgs = []
    features = []
    file_mapping = {i: f for i, f in enumerate(img_paths)}

    for i, f in enumerate(img_paths):
        if i == 20:
            break
        img = Image.open(f).convert('RGB')
        input_img = V(center_crop(img).unsqueeze(0))
        # imgs.append(input_img)
        feature = model.forward(input_img)
        features.append(feature)

    # logging.ingo('Images loaded and transformed...')
    # features = model.forward(imgs)
    end = time()
    logging.info('Generation time: %.2f s.' %(end-start))
    return features, file_mapping


def save_features(feat_fname, feat, map_fname, file_mapping):
    '''
    :param feat_fname: path to save 'feat' to
    :param feat: features (vector embeddings)
    :param map_fname:
    :param file_mapping:
    :return:
    '''
    logging.info('Saving features...')
    with open(feat_fname, 'wb') as f:
        pickle.dump(feat, f)
        f.close()
    with open(map_fname, 'wb') as f:
        pickle.dump(file_mapping, f)
        f.close()


def load_features(feat_fname, map_fname):
    """
    Loads features and file_item mapping from disk
    :param feat_fname: path to load features from
    :param map_fname: path to load mapping from
    :return: feature array and file_item mapping to disk
    """
    logging.info("Loading features...")
    with open(feat_fname, 'rb') as f:
        images_features = pickle.load(f)
        f.close()

    with open(map_fname, 'rb') as f:
        index_str = pickle.load(f)
        file_index = {int(k): str(v) for k, v in index_str.items()}
    return images_features, file_index


def index_features(features, n_trees=100, dims=512, is_dict=False):
    """
    Use Annoy to index our features to be able to query them rapidly
    :param features: array of item features
    :param n_trees: number of trees to use for Annoy. Higher is more precise but slower.
    :param dims: dimension of our features
    :return: an Annoy tree of indexed features
    """
    print ("Indexing features...")
    feature_index = AnnoyIndex(dims, metric='angular') # angular = cosine similarity
    for i, row in enumerate(features):
        vec = row.detach().cpu().numpy()
        print(vec.shape)
        if is_dict:
            vec = features[row].detach().cpu().numpy()
        feature_index.add_item(i, vec)
    feature_index.build(n_trees)
    return feature_index


# This is an inefficient implementation for the proof of context
def get_index(input_image, file_mapping):
    for index, file in file_mapping.items():
        if file == input_image:
            return index
    raise ValueError("Image %s not indexed" % input_image)


def search_index_by_key(key, feature_index, item_mapping, top_n=3):
    """
    Search an Annoy index by key, return n nearest items
    :param key: the index of our item in our array of features
    :param feature_index: an Annoy tree of indexed features
    :param item_mapping: mapping from indices to paths/names
    :param top_n: how many items to return
    :return: an array of [index, item, distance] of size top_n
    """
    distances = feature_index.get_nns_by_item(key, top_n, include_distances=True)
    return [[a, item_mapping[a], distances[1][i]] for i, a in enumerate(distances[0])]


def build_and_search(img_name, features, file_index):
    '''
    Pure image search
    :param features:
    :param file_index:
    :return:
    '''
    img = Image.open(img_name)
    input_image = V(center_crop(img).unsqueeze(0))
    image_index = index_features(features, dims=512)
    search_key = get_index(input_image, file_index)
    results = search_index_by_key(search_key, image_index, file_index)
    print(results)

def forward_pass_pipeline(arch, img_name):
    '''
    :param img_name: example: 'notebooks/data/forest path/49159594286.jpg'
    :return: top 5 probabilities and class labels
    '''
    img = Image.open(img_name)
    input_img = V(center_crop(img).unsqueeze(0))

    model = load_pretrained_model(arch)
    classes = load_classes()

    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    top5 = []
    print('{} prediction on {}'.format(arch, img_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        top5.append((probs[i], classes[idx[i]]))

    return tuple(top5)
