from flask import Flask, render_template, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os

# ----------------------------------
# CONFIGURACIÓN DE FLASK
# ----------------------------------
app = Flask(__name__)

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
# FUNCIÓN PARA CARGAR EL MODELO SOLO CUANDO SE NECESITE
# ----------------------------------
def load_model():
    """Carga el modelo solo cuando se necesita, para evitar exceso de RAM en Render."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'leaf_model.pth')

    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']

    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, class_names

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

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'No se pudo procesar la imagen: {str(e)}'})

    # Cargar el modelo en este punto (solo cuando se llama)
    model, class_names = load_model()

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    # Liberar memoria después de predecir
    del model
    torch.cuda.empty_cache()

    return jsonify({'prediction': label})

# ----------------------------------
# EJECUCIÓN LOCAL
# ----------------------------------

# if __name__ == '__main__':
#    app.run(debug=True)
