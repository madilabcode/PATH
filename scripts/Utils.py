
import sys
sys.path.append('../TransPath')
import torch.nn as nn
from TransPath.ctran import ctranspath
from torch.utils.data import DataLoader
import scanpy as sc
import torchvision.transforms as transforms
import torch
import os
from Datasets import roi_dataset

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

BATCH_SIZE = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


transform  = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean = MEAN, std = STD)
    ])

def embed_imgs(images):
  dataset = roi_dataset(images.detach().cpu().numpy())

  database_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
  model = ctranspath()
  model.head = nn.Identity()
  td = torch.load(r'../TransPath/ctranspath.pth')
  model.load_state_dict(td['model'], strict=True)
  model = model.to(device)
  embedding_list = []
  model.eval()
  with torch.no_grad():
    for batch in database_loader:
      embedding = model(batch.cuda())
      embedding_list.append(embedding)

  return torch.cat(embedding_list)

def process_samples_imgs(samples, obj):
    """
    Load each sample with image and spatial neighbor information.
    
    Parameters:
    -----------
    paths : list
        List of sample file paths
    base_dir : str
        Base directory containing the sample files
        
    Returns:
    --------
    images_list : list
        List of torch tensors containing image segments for each sample
    neighbors_list : list
        List of neighbor indices for each sample. Each sample contains a list where
        each element is an array of 31 closest neighbor indices for that spot.
    """
    
    spots = obj.obs.index.tolist()
    images_list = []
    idexs = []
    base_dir_df = obj.obs[["sample","base_dir"]].drop_duplicates().set_index("sample")
    data_set_df = obj.obs[["sample","dataset"]].drop_duplicates().set_index("sample")
    for idx, sample in enumerate(samples):
        try:
            base_dir = base_dir_df.loc[sample, "base_dir"]
            data_set = data_set_df.loc[sample, "dataset"]
            print("Processing sample with neighbors: ", sample)
            adata = sc.read_h5ad(os.path.join(base_dir, sample + ".h5ad"))
            slide_iamge = []
            img = list(adata.uns["spatial"].values())[0]["images"]["hires"]
            spatial_coords = adata.obsm['spatial']

            try:
                scale_factor = list(adata.uns["spatial"].values())[0]["scalefactors"]["tissue_hires_scalef"]
            except:
                scale_factor = 1

            scaled_coords = spatial_coords * scale_factor
    
            adata.obs.index = list(map(lambda x: str(x) + "_" + sample, adata.obs.index))
                        
            for idex, coord in enumerate(scaled_coords):
                if adata.obs.index[idex] in spots:
                    if "r" in adata.obs.columns:
                        r = adata.obs.iloc[idex]["r"]
                    else:
                        r = 100
                    idexs.append(adata.obs.iloc[idex].name)
                    segment = img[int(coord[1]-r*scale_factor) :int(coord[1]+r*scale_factor), 
                                int(coord[0]-r*scale_factor):int(coord[0]+r*scale_factor)]
                    segment = transform(segment)
                    segment = segment.unsqueeze(0)
                    slide_iamge.append(segment)

            images_list.append(torch.concat(slide_iamge))    

        
        except Exception as e:
            print(f"Error loading {sample}: {e}")
            # Add empty lists for failed samples
            images_list.append(torch.tensor([]))
        
    return images_list, idexs

def get_embeddings(model, dataloader: DataLoader) -> torch.Tensor:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch[0].to(device).float()
            embeddings.append(model.backbone(images).detach().cpu())
    return torch.concat(embeddings).numpy()