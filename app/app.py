from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io

# ----------------------------------
# CONFIGURACIÓN DE FLASK
# ----------------------------------
app = Flask(__name__)

# ----------------------------------
# CARGAR MODELO
# ----------------------------------
checkpoint = torch.load('../model/leaf_model.pth', map_location='cpu')
class_names = checkpoint['class_names']

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----------------------------------
# TRANSFORMACIÓN DE IMAGEN
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------------
# RUTA PRINCIPAL
# ----------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ----------------------------------
# RUTA DE PREDICCIÓN
# ----------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'})

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    return jsonify({'prediction': label})

# ----------------------------------
# EJECUCIÓN
# ----------------------------------
if __name__ == '__main__':
    app.run(debug=True)
