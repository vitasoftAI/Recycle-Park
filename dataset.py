import torch, os, numpy as np, pandas as pd, pickle
from glob import glob
from PIL import Image, ImageFile
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(2023)

class CustomDataset(Dataset):
    
    def __init__(self, root, data, lang, transformations = None, im_files = [".jpg", ".png", ".jpeg"]):
        super().__init__()
        self.im_paths = glob(f"{root}/{data}/*/*{[im_file for im_file in im_files]}")
        if lang == "en": print("Obtaining images from the folders...")
        elif lang == "ko": print("이미지 가져오는 중입니다...")
        
        self.cls_names = []
        for idx, im_path in enumerate(self.im_paths):
            cls_name = self.get_dir_name(im_path).split("/")[-1]
            if cls_name not in self.cls_names: self.cls_names.append(cls_name)
            
        self.classes_dict = {cls_name: i for cls_name, i in zip(self.cls_names, range(len(self.cls_names)))}
        self.transformations = transformations
        
    def __len__(self): return len(self.im_paths)

    def get_cls_len(self):
        
        di = {}
        for idx, im in enumerate(self.im_paths):
            cls_name = self.get_dir_name(im).split("/")[-1]
            if cls_name in di: di[cls_name] += 1
            else: di[cls_name] = 1
        
        im_count, threshold = 0, 30
        for cls_name, count in di.items():
            if count < threshold: im_count += 1
            print(f"Class {cls_name} has {count} images.")           
        print(f"\n{im_count} classes out of {len(di)} classes have less than {threshold} images.\n")
        
        return di
    
    def get_dir_name(self, path): return os.path.dirname(path)

    def get_im_label(self, path): return self.classes_dict[str(self.get_dir_name(path).split("/")[-1])]

    def get_cls_info(self): return list(self.classes_dict.keys()), len(self.classes_dict)
    
    def get_ims_paths(self, idx):
        
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

# class CustomDataset(Dataset):
    
#     def __init__(self, data, root, lang, transformations = None, threshold = 100, im_files = [".jpg", ".png", ".jpeg"]):
#         super().__init__()
        
#         therest = "/*/*/*/*/*"
#         data_path = f"{root}{data}{therest}"
#         excel_fname = "제네시스" if "genesis" in data else ("기아" if "kia" in data else "현대")
#         # self.all_classes = [cls_name for cls_name in pd.ExcelFile(classes_path).parse(0)["부품번호"]]
#         di = dict(pd.ExcelFile(f"excel_files/{excel_fname}.xlsx").parse(0))
#         new_di = {data: di["클래스별로"][idx] for idx, data in enumerate(di[f"{excel_fname}"])}
#         self.all_classes = [cls for cls, num in new_di.items() if num > threshold]
#         self.ims_paths = [im_path for im_path in glob(f"{data_path}/*[{im_file for im_file in im_files}]")]
#         self.ims = [im_path for im_path in self.ims_paths if self.get_dir_name(im_path).split("/")[-1] in self.all_classes]
        
#         self.cls_names = []
#         if lang == "en": print("Obtaining images from the folders...")
#         elif lang == "ko": print("이미지 가져오는 중입니다...")
#         for idx, im_path in enumerate(self.ims):
#             cls_name = self.get_dir_name(im_path).split("/")[-1]
#             if cls_name not in self.cls_names: self.cls_names.append(cls_name)
            
#         self.classes_dict = {cls_name: i for cls_name, i in zip(self.cls_names, range(len(self.cls_names)))}
#         self.transformations = transformations
        
#     def __len__(self): return len(self.ims)

#     def get_cls_len(self):
        
#         di = {}
#         for idx, im in enumerate(self.ims):
#             cls_name = self.get_dir_name(im).split("/")[-1]
#             if cls_name in di: di[cls_name] += 1
#             else: di[cls_name] = 1
        
#         im_count, threshold = 0, 30
#         for cls_name, count in di.items():
#             if count < threshold: im_count += 1
#             print(f"Class {cls_name} has {count} images.")           
#         print(f"\n{im_count} classes out of {len(di)} classes have less than {threshold} images.\n")
        
#         return di
    
#     def get_dir_name(self, path): return os.path.dirname(path)

#     def get_im_label(self, path): return self.classes_dict[str(self.get_dir_name(path).split("/")[-1])]

#     def get_cls_info(self): return list(self.classes_dict.keys()), len(self.classes_dict)
    
#     def get_ims_paths(self, idx):
        
#         qry_im_path = self.ims[idx]
#         qry_im_lbl = self.get_dir_name(qry_im_path).split("/")[-1]
        
#         pos_im_paths = [fname for fname in self.ims if qry_im_lbl in fname]
#         neg_im_paths = [fname for fname in self.ims if qry_im_lbl not in fname]
#         assert len(self.ims) == len(pos_im_paths) + len(neg_im_paths), "Please check the length of the data!"
        
#         pos_random_idx = int(np.random.randint(0, len(pos_im_paths), 1))
#         neg_random_idx = int(np.random.randint(0, len(neg_im_paths), 1))
        
#         pos_im_path = pos_im_paths[pos_random_idx]
#         neg_im_path = neg_im_paths[neg_random_idx]
        
#         qry_im_lbl = self.get_im_label(qry_im_path)
#         pos_im_lbl = self.get_im_label(pos_im_path)
#         neg_im_lbl = self.get_im_label(neg_im_path)
        
#         return qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl
    
#     def __getitem__(self, idx): 
        
#         qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl = self.get_ims_paths(idx)
        
#         qry_im, pos_im, neg_im = Image.open(qry_im_path), Image.open(pos_im_path), Image.open(neg_im_path)
        
#         if self.transformations is not None: qry_im, pos_im, neg_im = self.transformations(qry_im), self.transformations(pos_im), self.transformations(neg_im)
        
#         out_dict = {}
        
#         out_dict["qry_im"] = qry_im
#         out_dict["pos_im"] = pos_im
#         out_dict["neg_im"] = neg_im
        
#         out_dict["qry_im_lbl"] = qry_im_lbl
#         out_dict["pos_im_lbl"] = pos_im_lbl
#         out_dict["neg_im_lbl"] = neg_im_lbl
        
#         out_dict["qry_im_path"] = qry_im_path
#         out_dict["pos_im_path"] = pos_im_path
#         out_dict["neg_im_path"] = neg_im_path
        
#         return out_dict
    
# class CustomDataset(Dataset):
    
#     def __init__(self, data, transformations = None, threshold = 300, im_files = [".jpg", ".png", ".jpeg"]):
#         super().__init__()
        
#         classes_path = "/mnt/data/bekhzod/recycle_park/AI 학습 대상 부품번호.xls"
#         root, therest = "/mnt/data/bekhzod/recycle_park/", "/*/*/*/*/*"
#         data_path = f"{root}{data}{therest}"
#         excel_fname = "제네시스" if "genesis" in data else ("기아" if "kia" in data else "현대")
#         self.all_classes = [cls_name for cls_name in pd.ExcelFile(classes_path).parse(0)["부품번호"]]
#         # di = dict(pd.ExcelFile(f"excel_files/{excel_fname}.xlsx").parse(0))
#         # new_di = {data: di["클래스별로"][idx] for idx, data in enumerate(di[f"{excel_fname}"])}
#         # self.all_classes = [cls for cls, num in new_di.items() if num > threshold]
#         self.ims_paths = [im_path for im_path in glob(f"{data_path}/*[{im_file for im_file in im_files}]")]
#         self.ims = [im_path for im_path in self.ims_paths if self.get_dir_name(im_path).split("/")[-1] in self.all_classes]
        
#         self.cls_names = []
#         print("Obtaining images from the folders...")
#         for idx, im_path in enumerate(self.ims):
#             cls_name = self.get_dir_name(im_path).split("/")[-1]
#             if cls_name not in self.cls_names: self.cls_names.append(cls_name)
            
#         self.classes_dict = {cls_name: i for cls_name, i in zip(self.cls_names, range(len(self.cls_names)))}
#         self.transformations = transformations
        
#     def __len__(self): return len(self.ims)

#     def get_cls_len(self):
        
#         di = {}
#         for idx, im in enumerate(self.ims):
#             cls_name = self.get_dir_name(im).split("/")[-1]
#             if cls_name in di: di[cls_name] += 1
#             else: di[cls_name] = 1
        
#         im_count, threshold = 0, 30
#         for cls_name, count in di.items():
#             if count < threshold: im_count += 1
#             print(f"Class {cls_name} has {count} images.")           
#         print(f"\n{im_count} classes out of {len(di)} classes have less than {threshold} images.\n")
        
#         return di
    
#     def get_dir_name(self, path): return os.path.dirname(path)

#     def get_im_label(self, path): return self.classes_dict[str(self.get_dir_name(path).split("/")[-1])]

#     def get_cls_info(self): return list(self.classes_dict.keys()), len(self.classes_dict)
    
#     def get_ims_paths(self, idx):
        
#         qry_im_path = self.ims[idx]
#         qry_im_lbl = self.get_dir_name(qry_im_path).split("/")[-1]
        
#         pos_im_paths = [fname for fname in self.ims if qry_im_lbl in fname]
#         neg_im_paths = [fname for fname in self.ims if qry_im_lbl not in fname]
#         assert len(self.ims) == len(pos_im_paths) + len(neg_im_paths), "Please check the length of the data!"
        
#         pos_random_idx = int(np.random.randint(0, len(pos_im_paths), 1))
#         neg_random_idx = int(np.random.randint(0, len(neg_im_paths), 1))
        
#         pos_im_path = pos_im_paths[pos_random_idx]
#         neg_im_path = neg_im_paths[neg_random_idx]
        
#         qry_im_lbl = self.get_im_label(qry_im_path)
#         pos_im_lbl = self.get_im_label(pos_im_path)
#         neg_im_lbl = self.get_im_label(neg_im_path)
        
#         return qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl
    
#     def __getitem__(self, idx): 
        
#         qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl = self.get_ims_paths(idx)
        
#         qry_im, pos_im, neg_im = Image.open(qry_im_path), Image.open(pos_im_path), Image.open(neg_im_path)
        
#         if self.transformations is not None: qry_im, pos_im, neg_im = self.transformations(qry_im), self.transformations(pos_im), self.transformations(neg_im)
        
#         out_dict = {}
        
#         out_dict["qry_im"] = qry_im
#         out_dict["pos_im"] = pos_im
#         out_dict["neg_im"] = neg_im
        
#         out_dict["qry_im_lbl"] = qry_im_lbl
#         out_dict["pos_im_lbl"] = pos_im_lbl
#         out_dict["neg_im_lbl"] = neg_im_lbl
        
#         out_dict["qry_im_path"] = qry_im_path
#         out_dict["pos_im_path"] = pos_im_path
#         out_dict["neg_im_path"] = neg_im_path
        
#         return out_dict

# class CustomDataset(Dataset):
    
#     def __init__(self, data, transformations = None, im_files = [".jpg", ".png", ".jpeg"]):
#         super().__init__()
        
#         root, therest = "/mnt/data/bekhzod/recycle_park/", "/*/*/*/*/*"
#         data_path = f"{root}{data}{therest}"
#         with open('data_info/classes.pickle', 'rb') as handle: self.all_classes = pickle.load(handle)
#         self.ims_paths = [im_path for im_path in glob(f"{data_path}/*[{im_file for im_file in im_files}]")]
#         # self.ims = [im_path for im_path in self.ims_paths if self.get_dir_name(im_path).split("/")[-1] in self.all_classes]
#         self.ims = self.ims_paths
        
#         # print_info = {}
#         # for idx, im_path in enumerate(self.ims_paths):
#         #     cls_name = self.get_dir_name(im_path).split("/")[-1]
#         #     if cls_name not in print_info: print_info[cls_name] = False
#         #     if not print_info[cls_name]: print(cls_name); print_info[cls_name] = True
#         # print(len(print_info))
#         # print(len(self.all_classes))
#         # print(len(self.ims))
        
#         self.cls_names = []
#         print("Obtaining images from the folders...")
#         for idx, im_path in enumerate(self.ims):
#             cls_name = self.get_dir_name(im_path).split("/")[-1]
#             if cls_name not in self.cls_names: self.cls_names.append(cls_name)
            
#         self.classes_dict = {cls_name: i for cls_name, i in zip(self.cls_names, range(len(self.cls_names)))}
#         self.transformations = transformations
        
#     def __len__(self): return len(self.ims)

#     def get_cls_len(self):
        
#         di = {}
#         for idx, im in enumerate(self.ims):
#             cls_name = self.get_dir_name(im).split("/")[-1]
#             if cls_name in di: di[cls_name] += 1
#             else: di[cls_name] = 1
        
#         im_count, threshold = 0, 30
#         for cls_name, count in di.items():
#             if count < threshold: im_count += 1
#             print(f"Class {cls_name} has {count} images.")           
#         print(f"\n{im_count} classes out of {len(di)} classes have less than {threshold} images.\n")
        
#         return di
    
#     def get_dir_name(self, path): return os.path.dirname(path)

#     def get_im_label(self, path): return self.classes_dict[str(self.get_dir_name(path).split("/")[-1])]

#     def get_cls_info(self): return list(self.classes_dict.keys()), len(self.classes_dict)
    
#     def get_ims_paths(self, idx):
        
#         qry_im_path = self.ims[idx]
#         qry_im_lbl = self.get_dir_name(qry_im_path).split("/")[-1]
        
#         pos_im_paths = [fname for fname in self.ims if qry_im_lbl in fname]
#         neg_im_paths = [fname for fname in self.ims if qry_im_lbl not in fname]
#         assert len(self.ims) == len(pos_im_paths) + len(neg_im_paths), "Please check the length of the data!"
        
#         pos_random_idx = int(np.random.randint(0, len(pos_im_paths), 1))
#         neg_random_idx = int(np.random.randint(0, len(neg_im_paths), 1))
        
#         pos_im_path = pos_im_paths[pos_random_idx]
#         neg_im_path = neg_im_paths[neg_random_idx]
        
#         qry_im_lbl = self.get_im_label(qry_im_path)
#         pos_im_lbl = self.get_im_label(pos_im_path)
#         neg_im_lbl = self.get_im_label(neg_im_path)
        
#         return qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl
    
#     def __getitem__(self, idx): 
        
#         qry_im_path, qry_im_lbl, pos_im_path, pos_im_lbl, neg_im_path, neg_im_lbl = self.get_ims_paths(idx)
        
#         qry_im, pos_im, neg_im = Image.open(qry_im_path), Image.open(pos_im_path), Image.open(neg_im_path)
        
#         if self.transformations is not None: qry_im, pos_im, neg_im = self.transformations(qry_im), self.transformations(pos_im), self.transformations(neg_im)
        
#         out_dict = {}
        
#         out_dict["qry_im"] = qry_im
#         out_dict["pos_im"] = pos_im
#         out_dict["neg_im"] = neg_im
        
#         out_dict["qry_im_lbl"] = qry_im_lbl
#         out_dict["pos_im_lbl"] = pos_im_lbl
#         out_dict["neg_im_lbl"] = neg_im_lbl
        
#         out_dict["qry_im_path"] = qry_im_path
#         out_dict["pos_im_path"] = pos_im_path
#         out_dict["neg_im_path"] = neg_im_path
        
#         return out_dict
    
    
# class CustomClassificationDataset(Dataset):
    
#     def __init__(self, data, transformations = None, classes_path = "/mnt/data/bekhzod/recycle_park/AI 학습 대상 부품번호.xls", im_files = [".jpg", ".png", ".jpeg"]):
#         super().__init__()
        
#         root, therest = "/mnt/data/bekhzod/recycle_park/", "/*/*/*/*/*"
#         data_path = f"{root}{data}{therest}"
        
#         self.all_classes = [cls_name for cls_name in pd.ExcelFile(classes_path).parse(0)["부품번호"]]
#         self.ims_paths = [im_path for im_path in glob(f"{data_path}/*[{im_file for im_file in im_files}]")]
#         self.ims = [im_path for im_path in self.ims_paths if self.get_dir_name(im_path).split("/")[-1] in self.all_classes]
        
#         self.cls_names = []
#         print("Obtaining images from the folders...")
#         for idx, im_path in enumerate(self.ims):
#             cls_name = self.get_dir_name(im_path).split("/")[-1]
#             if cls_name not in self.cls_names: self.cls_names.append(cls_name)
            
#         self.classes_dict = {cls_name: i for cls_name, i in zip(self.cls_names, range(len(self.cls_names)))}
#         self.transformations = transformations
        
#     def __len__(self): return len(self.ims)

#     def get_cls_len(self):
        
#         di = {}
#         for idx, im in enumerate(self.ims):
#             cls_name = self.get_dir_name(im).split("/")[-1]
#             if cls_name in di: di[cls_name] += 1
#             else: di[cls_name] = 1
        
#         im_count, threshold = 0, 30
#         for cls_name, count in di.items():
#             if count < threshold: im_count += 1
#             print(f"Class {cls_name} has {count} images.")           
#         print(f"\n{im_count} classes out of {len(di)} classes have less than {threshold} images.\n")
        
#         return di
    
#     def get_dir_name(self, path): return os.path.dirname(path)

#     def get_im_label(self, path): return self.classes_dict[str(self.get_dir_name(path).split("/")[-1])]

#     def get_cls_info(self): return list(self.classes_dict.keys()), len(self.classes_dict)
    
#     def get_ims_paths(self, idx):
        
#         qry_im_path = self.ims[idx]
#         qry_im_lbl = self.get_dir_name(qry_im_path).split("/")[-1]
        
#         qry_im_lbl = self.get_im_label(qry_im_path)
        
#         return qry_im_path, qry_im_lbl
    
#     def __getitem__(self, idx): 
        
#         qry_im_path, qry_im_lbl = self.get_ims_paths(idx)
        
#         qry_im = Image.open(qry_im_path)
        
#         if self.transformations is not None: qry_im = self.transformations(qry_im)
        
#         return qry_im, qry_im_lbl
    
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