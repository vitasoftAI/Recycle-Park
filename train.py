# Import libraries
import torch, torchmetrics, wandb, timm, argparse, yaml, os, pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch import nn
from torchmetrics import F1Score, Precision, Accuracy
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as tfs
from dataset import CustomDataset, get_dls
from transformations import get_tfs
from utils import get_fm, makedirs

class CustomModel(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.
    
    Parameters:
    
        input_shape  - shape of input to the model, tuple -> int;
        model_name   - name of the model from timm library, str;
        num_classes  - number of classes to be outputed from the model, int;
        lr           - learning rate value, float.
    
    """
    
    def __init__(self, input_shape, model_name, num_classes, lr):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.pr = Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
        
        self.cos_loss = torch.nn.CosineEmbeddingLoss(margin = 0.3)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        
        self.cos = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
        
        self.lbls = {"cos_pos": torch.tensor(1.).unsqueeze(0), "cos_neg": torch.tensor(-1.).unsqueeze(0)}
        

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def forward(self, inp): return self.model(inp)
    
    def training_step(self, batch, batch_idx):
        
        qry_ims, pos_ims, neg_ims, qry_im_lbls, pos_im_lbls, neg_im_lbls = batch["qry_im"], batch["pos_im"], batch["neg_im"], batch["qry_im_lbl"], batch["pos_im_lbl"], batch["neg_im_lbl"]
        
        qry_fms = self.model.forward_features(qry_ims)
        pos_fms = self.model.forward_features(pos_ims)
        neg_fms = self.model.forward_features(neg_ims)
        
        pred_qry_lbls = self.model.forward_head(qry_fms)
        pred_pos_lbls = self.model.forward_head(pos_fms)
        pred_neg_lbls = self.model.forward_head(neg_fms)
        
        qry_fms, pos_fms, neg_fms = get_fm(qry_fms), get_fm(pos_fms), get_fm(neg_fms)
        
        cos_pos_loss = self.cos_loss(qry_fms, pos_fms, self.lbls["cos_pos"].to("cuda"))
        cos_neg_loss = self.cos_loss(qry_fms, neg_fms, self.lbls["cos_neg"].to("cuda"))
        cos_loss = cos_pos_loss + cos_neg_loss
        
        ce_qry_loss = self.ce_loss(pred_qry_lbls, qry_im_lbls)
        ce_poss_loss = self.ce_loss(pred_pos_lbls, pos_im_lbls)
        
        ce_loss = ce_qry_loss + ce_poss_loss
        
        loss = cos_loss + ce_loss
        
        # Train metrics
        qry_lbls = torch.argmax(pred_qry_lbls, dim = 1)
        acc = self.accuracy(qry_lbls, qry_im_lbls)
        pr = self.pr(qry_lbls, qry_im_lbls)
        f1 = self.f1(qry_lbls, qry_im_lbls)
        
#         top3, top1 = 0, 0
#         for idx, lbl_im in enumerate(qry_im_lbls):
            
#             cos_sim = self.cos(qry_fms[idx].unsqueeze(dim = 0), pos_fms)
            
#             vals, inds = torch.topk(cos_sim, k = 3)
            
#             if qry_im_lbls[idx] == qry_im_lbls[inds[0]] or qry_im_lbls[idx] == qry_im_lbls[inds[1]] or qry_im_lbls[idx] == qry_im_lbls[inds[2]]: top3 += 1
#             if qry_im_lbls[idx] in qry_im_lbls[inds[0]]: top1 += 1
        
        self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("train_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("train_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        qry_ims, pos_ims, neg_ims, qry_im_lbls, pos_im_lbls, neg_im_lbls = batch["qry_im"], batch["pos_im"], batch["neg_im"], batch["qry_im_lbl"], batch["pos_im_lbl"], batch["neg_im_lbl"]
        
        qry_fms = self.model.forward_features(qry_ims)
        pos_fms = self.model.forward_features(pos_ims)
        neg_fms = self.model.forward_features(neg_ims)
        
        pred_qry_lbls = self.model.forward_head(qry_fms)
        pred_pos_lbls = self.model.forward_head(pos_fms)
        pred_neg_lbls = self.model.forward_head(neg_fms)
        
        qry_fms, pos_fms, neg_fms = get_fm(qry_fms), get_fm(pos_fms), get_fm(neg_fms)
        
        cos_pos_loss = self.cos_loss(qry_fms, pos_fms, self.lbls["cos_pos"].to("cuda"))
        cos_neg_loss = self.cos_loss(qry_fms, neg_fms, self.lbls["cos_neg"].to("cuda"))
        
        cos_loss = cos_pos_loss + cos_neg_loss
        
        ce_qry_loss = self.ce_loss(pred_qry_lbls, qry_im_lbls)
        ce_poss_loss = self.ce_loss(pred_pos_lbls, pos_im_lbls)
        
        ce_loss = ce_qry_loss + ce_poss_loss
        
        loss = cos_loss + ce_loss
        
        # Train metrics
        qry_lbls = torch.argmax(pred_qry_lbls, dim = 1)
        acc = self.accuracy(qry_lbls, qry_im_lbls)
        pr = self.pr(qry_lbls, qry_im_lbls)
        f1 = self.f1(qry_lbls, qry_im_lbls)
        
#         top3, top1 = 0, 0
#         for idx, lbl_im in enumerate(qry_im_lbls):
            
#             cos_sim = self.cos(qry_fms[idx].unsqueeze(dim = 0), pos_fms)
#             vals, inds = torch.topk(cos_sim, k = 3)
            
#             if qry_im_lbls[idx] == qry_im_lbls[inds[0]] or qry_im_lbls[idx] == qry_im_lbls[inds[1]] or qry_im_lbls[idx] == qry_im_lbls[inds[2]]: top3 += 1
#             if qry_im_lbls[idx] in qry_im_lbls[inds[0]]: top1 += 1
        
        self.log("val_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("val_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        # self.log("val_top3", top3 / len(qry_im_lbls), on_step = False, on_epoch = True, logger = True, sync_dist = True)
        # self.log("val_top1", top1 / len(qry_im_lbls), on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("val_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("val_pr", pr, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        
        return loss
    
class ImagePredictionLogger(Callback):
    
    def __init__(self, val_samples, cls_names = None, num_samples = 8):
        super().__init__()
        self.num_samples, self.cls_names = num_samples, cls_names
        self.val_imgs, self.val_labels = val_samples["qry_im"], val_samples["qry_im_lbl"]
        
    def on_validation_epoch_end(self, trainer, pl_module):
        
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device = pl_module.device)
        val_labels = self.val_labels.to(device = pl_module.device)
        
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        
        # Log the images as wandb Image
        if self.cls_names != None:
            trainer.logger.experiment.log({
                "Sample Validation Prediction Results":[wandb.Image(x, caption = f"Predicted class: {self.cls_names[pred]}, Ground truth class: {self.cls_names[y]}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]})

def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    assert args.lang in ["ko", "en"], "Please choose either English or Korean! | 영어나 한국어를 선택하세요!"
    
    makedirs(args.save_model_path)
    makedirs(args.save_data_path)
    # Get train arguments    
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    if args.lang == "en": print(f"\nTraining Arguments:\n\n{argstr}")
    if args.lang == "ko": print(f"\n학습 과정 argument 명단:\n\n{argstr}")
    os.system(f"wandb login --relogin {args.wandb_key}")
    tr_tfs, te_tfs = get_tfs(args.inp_im_size)
    # threshold = 200 if "kia" in args.data else (300 if "hyundai" in args.data else 0)
    ds = CustomDataset(root = args.root, data = args.data, lang = args.lang, transformations = te_tfs)
    cls_names, num_classes = ds.get_cls_info()
    
    cls_names_file = f"{args.save_data_path}/{args.data}_cls_names_new_classes.pkl"
    with open(f"{cls_names_file}", "wb") as f: pickle.dump(cls_names, f)
            
    tr_dl, val_dl, test_dl = get_dls(ds = ds, lang = args.lang, bs = args.batch_size)
    torch.save(test_dl, f"{args.save_data_path}/{args.data}_test_dl_{args.batch_size}_new_classes.pth")

    # Samples required by the custom ImagePredictionLogger callback to log image predictions. 
    val_samples = next(iter(val_dl))
    # val_imgs, val_labels = val_samples["qry_im"], val_samples["qry_im_lbl"]

    model = CustomModel(input_shape = args.inp_im_size, model_name = args.model_name, num_classes = num_classes, lr = args.learning_rate) 

    # Initialize wandb logger
    wandb_logger = WandbLogger(project = f"{args.data}", job_type = "train", name = f"{args.model_name}_{args.data}_{args.batch_size}_{args.learning_rate}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs = args.epochs, accelerator="gpu", devices = args.devices, strategy = "ddp", 
                         logger = wandb_logger,
                         callbacks = [EarlyStopping(monitor = "val_loss", mode = "min", patience = 3), ImagePredictionLogger(val_samples, cls_names),
                                      ModelCheckpoint(monitor = "val_loss", dirpath = args.save_model_path, filename = f"{args.data}_best_model_{args.model_name}_new_classes")])

    # Train the model
    trainer.fit(model, tr_dl, val_dl)
    
    # # Test the model
    # trainer.test(dataloaders = test_dl)
    
    # Close wandb run
    wandb.finish()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Image Classification Training Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-da", "--data", type = str, default = "new_kia", help = "Data name") # new_hyundai genesis30_50
    parser.add_argument("-r", "--root", type = str, default = "path/to/data", help = "Data name")
    parser.add_argument("-bs", "--batch_size", type = int, default = 128, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 4, help = "Number of GPUs for training")
    parser.add_argument("-l", "--lang", type = str, default = "en", help = "Language to be used to run the code")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 3e-4, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 30, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = "saved_models", help = "Path to the directory to save a trained model")
    parser.add_argument("-wk", "--wandb_key", type = str, default = "api_key", help = "Wandb key can be obtained from wandb.ai")
    parser.add_argument("-sd", "--save_data_path", type = str, default = "saved_dls", help = "Path to the directory to save the dataset information")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
