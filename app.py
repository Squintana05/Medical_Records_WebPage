from flask import Flask, request, jsonify, render_template, send_from_directory
import google.generativeai as genai
from PIL import Image
import os
import easyocr
import numpy as np
import statistics
import cv2
import pytesseract
from skimage.filters import *
from skimage.filters.rank import *
from skimage.morphology import *
from unidecode import unidecode
import re
from spellchecker import SpellChecker
import supervision as sv

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Doctor')
def doctor_page():
    return render_template('Doctor.html')

@app.route('/list_uploaded_files')
def list_uploaded_files():
    try:
        uploaded_files = os.listdir(UPLOAD_FOLDER)
        return jsonify({'files': uploaded_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/statics/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/paciente')
def paciente_page():
    return render_template('paciente.html')


API_KEY = "AIzaSyAgn7LN435CT__10KlUhNZ2Y04G1HMuAvE"
genai.configure(api_key=API_KEY)
modelo = genai.GenerativeModel("gemini-1.5-pro-latest")
try:
    genai.configure(api_key=API_KEY)
    modelo = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    print(f"Error al configurar la API de Gemini: {str(e)}")
    modelo = None

@app.route("/consult_gemini", methods=["POST"])

def consult_gemini():
    data = request.json
    file_name = data.get("filename")
    user_prompt = data.get("prompt")

    if not file_name or not user_prompt:
        return jsonify({"response": "Debe seleccionar un archivo y escribir un prompt."}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    if not os.path.exists(file_path):
        return jsonify({"response": "El archivo seleccionado no existe."}), 400
    
    if not modelo:
        return jsonify({"response": "Error al conectar con el modelo de IA."}), 500
    
    response = API_requests(file_path, user_prompt)
    return jsonify({"response": response})

def API_requests(file_path, user_prompt):
    if file_path.endswith((".jpg", ".png")):
        contenido = Image.open(file_path)
        respuesta = modelo.generate_content([user_prompt, contenido])
        return respuesta.text if hasattr(respuesta, "text") else "No se pudo generar la respuesta."
    else:
        return "El archivo no es una imagen válida (.jpg o .png)."
    
    

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        
        img_type = request.form.get("tipo_imagen")
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        img_cv = cv2.imread(filepath)
        
        def OCR_new(im):
            labels_list = []
            spell = SpellChecker(language='es')
            reader = easyocr.Reader(['es'], gpu=True)
            result = reader.readtext(im, detail=1)

            def clean_text(text):
                text = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ.,;:!?()/-]', '', text) 
                return text.strip()

            labels, confidences = zip(*[
                (spell.correction(unidecode(text)) or text, confidence) for _, text, confidence in result])
            labels = [clean_text(label) for label in labels]
            labels_list.append(labels)

            xyxy = np.array([[int(min(pt[0] for pt in bbox)), int(min(pt[1] for pt in bbox)),
                            int(max(pt[0] for pt in bbox)), int(max(pt[1] for pt in bbox))]
                            for bbox, _, _ in result])

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=np.array(confidences),
                class_id=np.zeros(len(result), dtype=int))
            
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            annotated_image = box_annotator.annotate(scene=im, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            
            return labels, np.array(annotated_image)
        def method_just_graphs(filepath):
            img = cv2.imread(filepath)
            x, y ,_ = img.shape
            grid_size = (x//20,1)
            y_tils = np.linspace(0, img.shape[0], grid_size[0] + 1)
            x_tils = np.linspace(0, img.shape[1], grid_size[1] + 1)
            list_imgs_cut = []

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    img_cut = img[int(y_tils[i]): int(y_tils[i + 1]), int(x_tils[j]): int(x_tils[j + 1])]
                    list_imgs_cut.append((img_cut, (int(y_tils[i]), int(x_tils[j]))))  
            
            slices_with_graphs=[]       
            for s in range(len(list_imgs_cut)):
                slice = list_imgs_cut[s][0]
                hist = cv2.calcHist([slice], [0],None,[256],[0,256])
                cols = np.count_nonzero(hist)
                if cols > 200:
                    slices_with_graphs.append(list_imgs_cut[s][0])
                    
            if slices_with_graphs:
                final_img = np.vstack(slices_with_graphs) 
            else:
                final_img = np.zeros_like(img)  
            
            return np.array(final_img)
        def method_muchText_graphs(image_path):   
            def preprocess_image(image_path):
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img, gray

            def detect_text_regions(gray):
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 15, 10)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
                return morph

            def detect_graph_regions(gray):
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
                mask_graph = np.ones_like(gray) * 255
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(mask_graph, (x1, y1), (x2, y2), (0), thickness=1)
                return mask_graph

            def remove_thick_lines(graph_mask):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 
                clean_graph = cv2.morphologyEx(graph_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                return clean_graph

            def segment_text_and_graph(img, gray):
                text_mask = detect_text_regions(gray)
                graph_mask = detect_graph_regions(gray)
                clean_graph = remove_thick_lines(graph_mask)
                text_only = cv2.bitwise_and(img, img, mask=text_mask)
                graph_only = cv2.bitwise_and(img, img, mask=clean_graph)
                return text_only, graph_only

            def main(image_path):
                img, gray = preprocess_image(image_path)
                _, graph_img = segment_text_and_graph(img, gray)
                return np.array(graph_img)
            
            img = main(image_path)    
            x, y, _ = img.shape
            grid_size = (x//20,1)
            y_tils = np.linspace(0, img.shape[0], grid_size[0] + 1)
            x_tils = np.linspace(0, img.shape[1], grid_size[1] + 1)
            list_imgs_cut = []

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    img_cut = img[int(y_tils[i]): int(y_tils[i + 1]), int(x_tils[j]): int(x_tils[j + 1])]
                    list_imgs_cut.append((img_cut, (int(y_tils[i]), int(x_tils[j]))))  
            
            slices_with_graphs,l_hists_zeros,l_cols=[],[],[]
            for img_c_s, _ in list_imgs_cut: 
                hist = cv2.calcHist([img_c_s], [0], None, [256], [0, 256])
                cols = np.count_nonzero(hist)
                l_hists_zeros.append(hist[0,0])
                l_cols.append(cols)
            
            for img_c_s, _ in list_imgs_cut: 
                hist = cv2.calcHist([img_c_s], [0], None, [256], [0, 256])
                cols = np.count_nonzero(hist)   
                if cols > int(statistics.mean(l_cols)) and hist[0, 0] < int(statistics.mean(l_hists_zeros)): 
                    text = pytesseract.image_to_string(img_c_s).split()
                        
                    if not any(word in text for word in ['CONCLUSION', 'conclusion', 'RESULTADOS']):  
                        slices_with_graphs.append(img_c_s)
                
            if slices_with_graphs:
                final_img = np.vstack(slices_with_graphs) 
            else:
                final_img = np.zeros_like(img)
            
            return final_img
        
        if img_type == 'Solo_texto':
            texto, img_detection = OCR_new(img_cv)
        
        elif img_type == 'Solo_grafica':
            texto,_ = OCR_new(img_cv)
            img_detection = method_just_graphs(filepath)
            
        elif img_type == 'Texto_y_grafica':
            texto, _ = OCR_new(img_cv)
            img_detection = method_muchText_graphs(filepath)
        else:
            return jsonify({'error': 'Tipo de imagen no válido'}), 400   
                    
        filename_without_extension, extension = os.path.splitext(file.filename)
        processed_filename = os.path.join(PROCESSED_FOLDER, f"processed_{filename_without_extension}{extension}")

        cv2.imwrite(processed_filename, cv2.cvtColor(img_detection, cv2.COLOR_RGB2BGR))
        
        return render_template("paciente.html", texto=texto, detected_image=processed_filename.replace("static/processed/", ""))


@app.route('/static/processed/<filename>')
def processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render asigna un puerto dinámico
    app.run(host="0.0.0.0", port=port)