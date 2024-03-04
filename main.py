from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os, time
from maskgen import large_lip_logic, medium_lip_logic, light_lip_logic

app = Flask(__name__)

output_directory = "./recieved_files/"
# print(large_lip_logic("images/1.png"))
pod_id_bkp = "a3juid7agss9ns"
@app.route('/large_lip_mask', methods=['POST'])
def large_lip():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    pod_id = request.form.get('pod_id', pod_id_bkp)
    if file:
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{filename.split('.')[0]}-{int(time.time())}.{file_extension}"
        temp_path = os.path.join(output_directory, unique_filename)
        file.save(temp_path)

        # result_path = large_lip_logic(temp_path,pod_id)
        b64 = large_lip_logic(temp_path,pod_id)
        return jsonify({'base64_data': b64})

        # return send_file(result_path, as_attachment=True)

@app.route('/medium_lip_mask', methods=['POST'])
def medium_lip():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    pod_id = request.form.get('pod_id', pod_id_bkp)
    if file:
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{filename.split('.')[0]}-{int(time.time())}.{file_extension}"
        temp_path = os.path.join(output_directory, unique_filename)
        file.save(temp_path)

        # result_path = medium_lip_logic(temp_path)
        b64 = medium_lip_logic(temp_path,pod_id)
        return jsonify({'base64_data': b64})
        # return send_file(result_path, as_attachment=True)

@app.route('/light_lip_mask', methods=['POST'])
def light_lip():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    pod_id = request.form.get('pod_id', pod_id_bkp)
    
    if file:
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{filename.split('.')[0]}-{int(time.time())}.{file_extension}"
        temp_path = os.path.join(output_directory, unique_filename)
        file.save(temp_path)

        # result_path = light_lip_logic(temp_path)

        b64 = light_lip_logic(temp_path,pod_id)
        return jsonify({'base64_data': b64})
        # return send_file(result_path, as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=True,port=8181,host= '0.0.0.0')
