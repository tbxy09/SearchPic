from flask import request
from model import get_prediction
import jsonify

@app.route('/predict',methods=['POST'])
def predict():
    if request.methods='POST':
        file = request.files[file]
        # img reader
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id':class_id, 'class_name':class_name})