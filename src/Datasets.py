
import sys
from torch.utils.data import Dataset
import torch
import os
import random
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)



class GeneExpressionDataset(Dataset):
    def __init__(self, gene_expression, slides):
        self.gene_expression = gene_expression
        self.slides = slides
        self.index_list = list(range(len(self.gene_expression)))

    def __len__(self):
        return len(self.gene_expression)

    def __getitem__(self, idx):
        return self.gene_expression[idx], self.slides[idx]


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.index_list = list(range(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

class roi_dataset(Dataset):
    def __init__(self, images ):
        super().__init__()
    
        self.images_lst = images

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        return  self.images_lst[idx]


class roi_ge_dataset(Dataset):
    def __init__(self, images, kegg_expression, slide_number, sample_number, dataset_number):
        self.images = images
        self.kegg_expression = kegg_expression
        self.slide_number = slide_number
        self.sample_number = sample_number
        self.dataset_number = dataset_number
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.kegg_expression[idx], self.slide_number[idx], self.sample_number[idx], self.dataset_number[idx]