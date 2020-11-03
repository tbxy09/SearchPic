import io
import json
import os
from flask import Flask, jsonify, request
import torch
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from PIL import Image
import cv2

# homeprod_classes = ['waterheater', 'ricecooker', 'airclean', 'airwarterheater', 'Juicecup', 'electricfan', 'lunchbox', 'hotpot', 'washingmachine','humidifier','Electrickettle']
homeprod_class_index = json.load(open('homeprod_classes_index.json'))
# app = Flask(__name__)
app = Flask(__name__, static_folder='../dist', static_url_path='/')
cfg = get_cfg()
cfg.merge_from_file("/home/tbxy09/projects/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.OUTPUT_DIR = "/d2/p2/tbxy09/dataset_10_26_17_43"
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,'model_final.pth')
classes = homeprod_class_index
# print(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

predictor = DefaultPredictor(cfg)
# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

def get_prediction(image):
    # detectron2 predictor 引入 cfg 文件 yaml
    print('enter get_prediction')
    prediction = predictor(image)
    predictions = prediction['instances'].to(torch.device('cpu'))
    print(predictions)
    if len(predictions)>0:
        pred_classes = predictions.pred_classes if predictions.has("pred_classes") else None
        pred_scores = predictions.scores if predictions.has("scores") else None
        pred_scores = pred_scores.numpy()
        pred_classes = pred_classes.numpy()
        idmax = pred_classes[pred_scores.argmax()]
        idmax = str(idmax)

        # electricfan not in tranning list
        # homeprod_class_index.remove['electricfan']
        print(homeprod_class_index)
        class_name = homeprod_class_index[idmax][0]
        class_id = homeprod_class_index[idmax][1]
    # class_id = [homeprod_class_index[str(c)][1] for c in [out.index(out.max())]]
        return class_id, class_name, pred_scores.max()
    else:
        return "None", "None", None

def _apply_exif_orientation(image):
    """
    util._apply_exif_orientation
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        print('get request',request.files.values())
        file = request.files['file']
        # name = request.files['name']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        image = _apply_exif_orientation(image)
        format = 'BGR'
        image = utils.convert_PIL_to_numpy(image, format)
        cv2.imwrite("test.jpg",image)
        # img = utils.read_image(name,format="BGR")
        #需要decode吗？
        class_id,class_name,prob = get_prediction(image=image) 
        
        return jsonify({'class_id':class_id,'class_name':class_name,'prob':str(prob)})
        # return jsonify({'ret':request.form.to_dict()})
@app.route('/')
def index():
        return app.send_static_file('index.html')
if __name__ == "__main__":
    app.run()
    
    # app = webapp2.WSGIApplication([
    #     ('/share-video', ShareVideo),
    # ], debug=True)