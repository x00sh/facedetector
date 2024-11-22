from flask import Flask, request, render_template_string, redirect
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'JPG'}

# if there's no upload folder, create it
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if there's a file part in the request
        if 'file' not in request.files:
            return redirect(request.url) # if there's no file part, redirect back to the request
        file = request.files['file']
        # if no file is selected, browser submits an empty part without the filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                prediction = predict_image(file_path)
                return render_template_string('''<!doctype html>
                    <title>Upload new File</title>
                    <h1>Uploaded File Prediction: {{ prediction }}</h1>
                    <a href="/">Upload another file</a>
                ''', prediction=prediction)
            except Exception as e:
                return str(e)
        else:
            return 'File not allowed'
    
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload a picture file</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def predict_image(image_path):
    # define the path to the state dictionary file 
    state_dict_path = "C://Users//zeesh//Downloads//FINAL CODE SUBMISSION//FINAL_MODEL_FYP.pth"

    # define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size of ResNet18
        transforms.ToTensor(),  # Convert image to tensor
    ])

    class FaceClassifier(nn.Module):
        def __init__(self):
            super(FaceClassifier, self).__init__()
            self.features = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.features.fc = nn.Linear(512, 2)

        def forward(self, x):
            x = self.features(x)
            return x

    # Initialize the model
    model = FaceClassifier()

    # Determine the device and ensure the model is on the right device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    # Open and transform the image, then load it to the same device as the model
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    
    # Perform model evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    labels = {0: 'face', 1: 'non-face'}
    return labels[predicted.item()]

if __name__ == '__main__':
    app.run(debug=True)