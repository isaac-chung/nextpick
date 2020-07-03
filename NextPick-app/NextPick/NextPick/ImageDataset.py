import os
import pandas as pd
from PIL import Image
from torchvision import transforms as trn
from torch.utils.data import Dataset, DataLoader

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


def load_data_paths(folder):
    class_names = [fold for fold in os.listdir(folder)]  # this should get folder names at the 'abbey' level
    labels_list = []
    names_list = []
    sub_paths = []

    for cl in class_names:
        if 'git' in cl:
            continue
        # skip all .pkl files
        img_files = [f for f in os.listdir(os.path.join(folder, cl)) if '.pkl' not in f]
        for img in img_files:
            labels_list.append(cl)
            names_list.append(img)
            sub_paths.append('/' + os.path.join(cl, img))

    df = pd.DataFrame(columns=['label', 'name', 'sub_paths'])
    df['label'] = labels_list
    df['name'] = names_list
    df['sub_paths'] = sub_paths
    return df
