import os, torch, argparse, pickle
from utils import get_state_dict, get_model, get_fm, create_dbs, predict_classes

def run(args):
    
    assert args.lang in ["ko", "en"], "Please choose either English or Korean! | 영어나 한국어를 선택하세요!"
    
    data_path = args.saved_data_path + args.data_name + "_test_dl_32_new_classes.pth"
    cls_path = args.saved_classes_path + args.data_name + "_cls_names_new_classes.pkl"
    test_dl = torch.load(data_path)
    with open(cls_path, "rb") as f: cls_names = pickle.load(f)
    if args.lang == "ko":
        print(f"테스트 데이셋에 {len(test_dl)}개의 배치가 있습니다!")
        print(f"테스트 데이셋에 {len(cls_names)}개의 파트번호가 있습니다!")
    elif args.lang == "en":
        print(f"There are {len(test_dl)} batches in the test dataloader!")
        print(f"There are {len(cls_names)} classes in the test dataloader!")
    
    model_path = args.saved_model_path + args.data_name + "_best_model_" + args.model_name + "_new_classes.ckpt"
    
    model = get_model(model_name = args.model_name, n_cls = len(cls_names), lang = args.lang,
                      device = args.device, saved_model_path = model_path)
    
    ims_all, qry_fms_all, pos_fms_all, im_lbls = create_dbs(model = model, test_dl = test_dl, model_name = args.model_name, lang = args.lang,
                                                 data_name = args.data_name, device = args.device, save_path = "saved_dbs")
    
    predict_classes(qry_fms_all, pos_fms_all, cls_names, im_lbls, num_top_stop = 5, top_k = 5)
        

if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Image Classification Inference Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-d", "--device", type = str, default = "cuda:1", help = "GPU device name")
    parser.add_argument("-dn", "--data_name", type = str, default = "genesis30_50", help = "Dataset name")
    parser.add_argument("-mn", "--model_name", type = str, default = 'rexnet_150', help = "Model name for backbone")
    parser.add_argument("-l", "--lang", type = str, default = "ko", help = "Language to be used to run the code")
    parser.add_argument("-sm", "--saved_model_path", type = str, default = "saved_models/", help = "Path to the directory with the trained model")
    parser.add_argument("-sd", "--saved_data_path", type = str, default = "saved_dls/", help = "Path to the directory with the saved dataloader")
    parser.add_argument("-sc", "--saved_classes_path", type = str, default = "saved_dls/", help = "Path to the directory with dataset class names")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
