# Import libraries
import io, json, timm, base64, torch, argparse, pickle
from torchvision import models
from PIL import Image
from flask import Flask, jsonify, request
from transformations import get_tfs
from gradio_demo_gen import load_model

# Initialize flask
app = Flask(__name__)

def run(args):

    with open(f"saved_dls/{args.data}_cls_names_new_classes.pkl", "rb") as f: cls_names = pickle.load(f)
    tfs = get_tfs((224, 224))[1]
    model = load_model(args.model_name, len(cls_names), f"saved_models/{args.data}_best_model_rexnet_150_new_classes.ckpt")

    @app.route('/predict', methods=['POST'])
    def predict():
        im = json.loads(request.data)['im']
        jpg_original = base64.b64decode(im)

        results = get_prediction(model = model, cls_names = cls_names, image_bytes=jpg_original)

        return jsonify({"results": results})

    def get_prediction(model, image_bytes, cls_names):

        results = {}
        im = tfs(Image.open(io.BytesIO(image_bytes)))
        pred = torch.nn.functional.softmax(model(im.unsqueeze(0).data), dim = 1)

        vals, inds = torch.topk(pred, k = 5)
        vals, inds = vals.squeeze(0), inds.squeeze(0)

        for idx, (val, ind) in enumerate(zip(vals, inds)):
            results[f"top_{idx + 1}"] = cls_names[ind.item()]

        return results

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Object Classification Demo")
    
    # Add arguments
    parser.add_argument("-dt", "--data", type = str, default = "genesis30_50", help = "Dataset name")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
    
    app.run(host='0.0.0.0', debug=False, port=8610)
    
