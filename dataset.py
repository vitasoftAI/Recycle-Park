# Import libraries
import torch, os, numpy as np, pandas as pd, pickle
from glob import glob; from PIL import Image, ImageFile
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T

# Option to get large image files
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Set manual seed
torch.manual_seed(2023)

class CustomDataset(Dataset):

    """
    
    This class gets several parameters and returns dataset to be trained.

    Parameters:

        root             - path to data with images, str;
        data             - data type, str;
        lang             - language to print, str;
        transformations  - transformations to be applied, transforms object;
        im_files         - image extension files, list -> str.    
    
    """
    
    def __init__(self, root, data, lang, transformations = None, im_files = [".jpg", ".png", ".jpeg"]):
        super().__init__()
        self.im_paths = glob(f"{root}/{data}/*/*{[im_file for im_file in im_files]}")
        if lang == "en": print("Obtaining images from the folders...")
        elif lang == "ko": print("이미지 가져오는 중입니다...")

        # Set list to get class names
        self.cls_names = []
        # Go through every image path
        for idx, im_path in enumerate(self.im_paths):
            # Get the class name
            cls_name = self.get_dir_name(im_path).split("/")[-1]
            if cls_name not in self.cls_names: self.cls_names.append(cls_name)
            
        # Create class names dictionary
        self.classes_dict = {cls_name: i for cls_name, i in zip(self.cls_names, range(len(self.cls_names)))}
        # Set the transformations to be applied
        self.transformations = transformations
        
    # Get the number of images in the dataset 
    def __len__(self): return len(self.im_paths)

    def get_cls_len(self):

        """

        This class return dictionary with class names.
        
        """
        
        # Set the dictionary
        di = {}
        # Go through every image path
        for idx, im in enumerate(self.im_paths):
            # Get the class name based on the image path
            cls_name = self.get_dir_name(im).split("/")[-1]
            
            # Add the class name to the dictionary
            if cls_name in di: di[cls_name] += 1
            else: di[cls_name] = 1
        
        # Set the values for the image count and threshold
        im_count, threshold = 0, 30
        
        for cls_name, count in di.items():
            if count < threshold: im_count += 1
            print(f"Class {cls_name} has {count} images.")           
        print(f"\n{im_count} classes out of {len(di)} classes have less than {threshold} images.\n")
        
        return di
    
    # This function gets directory name based on the image path
    def get_dir_name(self, path): return os.path.dirname(path)

    # This function gets image label based on the image path
    def get_im_label(self, path): return self.classes_dict[str(self.get_dir_name(path).split("/")[-1])]

    # This function gets information based on the classes dictionary 
    def get_cls_info(self): return list(self.classes_dict.keys()), len(self.classes_dict)
    
    def get_ims_paths(self, idx):

        """

        This function gets an index and returns meta data to train an AI model.

        Parameter:

            idx              - index, int.

        Outputs:

            qry_im_path      - image path of an query image, str;
            qry_im_lbl       - class label of an query image, int;
            pos_im_path      - image path of a positive image, str;
            pos_im_lbl       - class label of a positive image, int;
            neg_im_path      - image path of a negative image, str;
            neg_im_lbl       - class label of a negative image, int;        
        
        """
        
        qry_im_path = self.im_paths[idx]
        qry_im_lbl = self.get_dir_name(qry_im_path).split("/")[-1]
        
        pos_im_paths = [fname for fname in self.im_paths if qry_im_lbl in fname]
        neg_im_paths = [fname for fname in self.im_paths if qry_im_lbl not in fname]
        assert len(self.im_paths) == len(pos_im_paths) + len(neg_im_paths), "Please check the length of the data!"
        
        pos_random_idx = int(np.random.randint(0, len(pos_im_paths), 1))
        neg_random_idx = int(np.random.randint(0, len(neg_im_paths), 1))
        
        pos_im_path = pos_im_paths[pos_random_idx]
        neg_im_path = neg_im_paths[neg_random_idx]
        
        qry_im_lbl = self.get_im_label(qry_im_path)
        pos_im_lbl = self.get_im_label(pos_im_path)
        neg_im_lbl = self.get_im_label(neg_im_path)
        
        return qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl
    
    def __getitem__(self, idx): 
        
        qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl = self.get_ims_paths(idx)
        
        qry_im, pos_im, neg_im = Image.open(qry_im_path), Image.open(pos_im_path), Image.open(neg_im_path)
        
        if self.transformations is not None: qry_im, pos_im, neg_im = self.transformations(qry_im), self.transformations(pos_im), self.transformations(neg_im)
        
        out_dict = {}
        
        out_dict["qry_im"] = qry_im
        out_dict["pos_im"] = pos_im
        out_dict["neg_im"] = neg_im
        
        out_dict["qry_im_lbl"] = qry_im_lbl
        out_dict["pos_im_lbl"] = pos_im_lbl
        out_dict["neg_im_lbl"] = neg_im_lbl
        
        out_dict["qry_im_path"] = qry_im_path
        out_dict["pos_im_path"] = pos_im_path
        out_dict["neg_im_path"] = neg_im_path
        
        return out_dict
    
class InferenceCustomDataset(Dataset):
    
    def __init__(self, root, data, transformations = None, im_files = [".jpg", ".png", ".jpeg"]):
        super().__init__()
        
        self.transformations = transformations
        self.im_paths = glob(f"{root}/{data}/*/*/*/*/*/*{[im_file for im_file in im_files]}")
        self.im_paths = [im_path for im_path in self.im_paths if os.path.dirname(im_path).split("/")[-1] == "A"]
        
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im = Image.open(self.im_paths[idx]).convert("RGB")
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, self.im_paths[idx]
    
def get_dls(ds, lang, bs):
    
    # Get length of the dataset
    ds_len = len(ds)

    # Get length for train and validation datasets
    tr_len, val_len = int(ds_len * 0.8), int(ds_len * 0.1) 

    # Split the dataset into train, validation, and test datasets
    tr_ds, val_ds, test_ds = random_split(ds, [tr_len, val_len, ds_len - (tr_len + val_len)])

    if lang == "en":
        print(f"Number of train set images: {len(tr_ds)}")
        print(f"Number of validation set images: {len(val_ds)}")
        print(f"Number of test set images: {len(test_ds)}\n")
        
    elif lang == "ko":
        print(f"Train 데이터셋 이미지 수: {len(tr_ds)}")
        print(f"Validation 데이터셋 이미지 수: {len(val_ds)}")
        print(f"Test 데이터셋 이미지 수: {len(test_ds)}\n")

    tr_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = 8)
    val_dl = DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = 8)
    test_dl = DataLoader(test_ds, batch_size = bs, shuffle = False, num_workers = 8)

    return tr_dl, val_dl, test_dl  
