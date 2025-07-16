# 🍫 Sistema de Diagnóstico de Enfermedades en Cacao con Deep Learning

## 📋 Descripción

Este proyecto implementa un sistema automático de diagnóstico de enfermedades del cacao utilizando redes neuronales convolucionales (CNN) y técnicas de Deep Learning. El sistema puede identificar tres estados fitosanitarios:

- 🌱 **Sano** (healthy)
- 🦠 **Pudrición Negra** (black pod rot - Phytophthora)
- 🐛 **Barrenador de Mazorca** (pod borer - Conopomorpha)

## 🚀 Características

- ✅ Implementación de 4 arquitecturas CNN diferentes
- ✅ Transfer Learning con MobileNetV2 y ResNet50
- ✅ Validación estadística con prueba de McNemar
- ✅ Métricas robustas (MCC, F1-score, especificidad)
- ✅ Interfaz web con Streamlit
- ✅ Generación de reportes PDF
- ✅ Mapas de calor Grad-CAM para interpretabilidad

## 🛠️ Tecnologías Utilizadas

- **Python** 3.11
- **TensorFlow** 2.13.0
- **Keras** 2.13.0
- **Streamlit** 1.34.0
- **Google Colab** (para entrenamiento)
- **Kaggle API** 1.6.12

## Hardware del Proyecto:
- **GPU:** NVIDIA GeForce RTX 4060 Laptop
- **CPU:** AMD Ryzen 7 6800H
- **RAM:** 16GB DDR5-4800 MHz
- **Cuantización:** TensorRT 8.6.1

## 📊 Resultados

| Modelo | MCC | Especificidad | F1-Score |
|--------|-----|---------------|----------|
| CNN Simple | 0.744 | 0.915 | 0.830 |
| CNN Profunda | 0.727 | 0.910 | 0.812 |
| MobileNetV2 | **0.910** | **0.969** | **0.942** |
| ResNet50 | 0.910 | 0.97 | 0.89 |

## 🔧 Instalación y Configuración

### Paso 1: Crear Repositorio en GitHub

1. 📂 Ve a [GitHub](https://github.com) e inicia sesión
2. ➕ Haz clic en "New" para crear un nuevo repositorio
3. 📝 Nombra tu repositorio (ej: `cacao-disease-detection`)
4. ✅ Marca "Add a README file"
5. 🔘 Selecciona "Public" o "Private" según prefieras
6. 🎯 Haz clic en "Create repository"

### Paso 2: Clonar y Configurar el Repositorio

```bash
git clone https://github.com/tu-usuario/cacao-disease-detection.git
cd cacao-disease-detection
```

### Paso 3: Subir el Archivo CacaoDeep.ipynb

1. 📁 Coloca el archivo `CacaoDeep.ipynb` en la raíz del repositorio
2. 📤 Sube el archivo:
```bash
git add CacaoDeep.ipynb
git commit -m "Añadir notebook principal CacaoDeep"
git push origin main
```

## 🚀 Ejecución en Google Colab

### Preparación Inicial

1. 🌐 Abre [Google Colab](https://colab.research.google.com/)
2. 📂 Sube el archivo `CacaoDeep.ipynb` desde tu repositorio
3. 🔄 Asegúrate de tener una cuenta de Kaggle activa

### Ejecución Paso a Paso

#### 📦 **Celda 1: Instalación de Librerías**
```python
# Instala librerías requeridas
!pip install tensorflow scikit-learn matplotlib seaborn statsmodels kaggle opencv-python-headless tf-keras-vis --quiet
!pip install streamlit pdfkit
!apt-get install -y wkhtmltopdf
```
▶️ **Ejecutar**: Instala todas las dependencias necesarias para el proyecto

---

#### 🔑 **Celda 2: Configuración de Kaggle**
```python
from google.colab import files
files.upload()  # Subir kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
▶️ **Ejecutar**: 
1. 📁 Descarga tu archivo `kaggle.json` desde tu cuenta de Kaggle
2. 📤 Súbelo cuando se solicite
3. ✅ Configura las credenciales automáticamente

---

#### 📥 **Celda 3: Descarga del Dataset**
```python
# Descarga el dataset
!kaggle datasets download -d zaldyjr/cacao-diseases
!unzip -q cacao-diseases.zip -d cacao_data
```
▶️ **Ejecutar**: Descarga y descomprime el Cocoa Diseases Dataset (4,390 imágenes)

---

#### 🔍 **Celda 4: Exploración del Dataset**
```python
import os
base_dir = 'cacao_data/cacao_diseases/cacao_photos'
clases = os.listdir(base_dir)
print("Clases encontradas:", clases)
for clase in clases:
    ruta = os.path.join(base_dir, clase)
    print(f"{clase}: {len(os.listdir(ruta))} imágenes")
```
▶️ **Ejecutar**: Explora la estructura del dataset y cuenta las imágenes por clase

---

#### 🎯 **Celda 5: Balanceo y División del Dataset**
```python
import pandas as pd
import random

base_dir = 'cacao_data/cacao_diseases/cacao_photos'
clases = ['healthy', 'black_pod_rot', 'pod_borer']
imgs_per_class = 103

# Recolectar imágenes balanceadas
data = []
for clase in clases:
    ruta = os.path.join(base_dir, clase)
    imgs = os.listdir(ruta)
    # Filtrar solo archivos de imagen válidos
    imgs = [img for img in imgs if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Asegurarse de que hay suficientes imágenes para seleccionar
    if len(imgs) < imgs_per_class:
        print(f"Advertencia: La clase '{clase}' tiene menos de {imgs_per_class} imágenes. Usando todas las disponibles.")
        selected_imgs = imgs
    else:
        selected_imgs = random.sample(imgs, imgs_per_class)
    for img in selected_imgs:
        data.append({'filepath': os.path.join(ruta, img), 'label': clase})

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Mezclar

# Codificar etiquetas numéricas
label_dict = {clase: idx for idx, clase in enumerate(clases)}
df['label_num'] = df['label'].map(label_dict)

# Separar por clase
df_train, df_val, df_test = [], [], []
# Calcular el número de imágenes por conjunto para cada clase
# Asumiendo 103 imágenes por clase:
# 70% para entrenamiento (~72 imágenes)
# 15% para validación (~15 imágenes)
# 15% para prueba (~16 imágenes)
train_count = int(imgs_per_class * 0.70)
val_count = int(imgs_per_class * 0.15)
test_count = imgs_per_class - train_count - val_count # El resto para asegurar 103

for clase in clases:
    dft = df[df['label'] == clase]
    train = dft.iloc[:train_count]
    val = dft.iloc[train_count : train_count + val_count]
    test = dft.iloc[train_count + val_count : train_count + val_count + test_count] # Asegurar que no exceda el límite

    df_train.append(train)
    df_val.append(val)
    df_test.append(test)

df_train = pd.concat(df_train).sample(frac=1, random_state=42).reset_index(drop=True)
df_val = pd.concat(df_val).sample(frac=1, random_state=42).reset_index(drop=True)
df_test = pd.concat(df_test).sample(frac=1, random_state=42).reset_index(drop=True)

print("Train:", df_train['label'].value_counts())
print("Val:", df_val['label'].value_counts())
print("Test:", df_test['label'].value_counts())
```
▶️ **Ejecutar**: Balancea el dataset y lo divide en conjuntos de entrenamiento, validación y prueba

---

#### 🖼️ **Celda 6: Carga y Preprocesamiento de Imágenes**
```python
from PIL import Image
import numpy as np

IMG_SIZE = (128, 128)

def load_images(df):
    X = []
    y = []
    for i, row in df.iterrows():
        img = Image.open(row['filepath']).convert('RGB').resize(IMG_SIZE)
        X.append(np.array(img)/255.0)
        y.append(row['label_num'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

X_train, y_train = load_images(df_train)
X_val, y_val = load_images(df_val)
X_test, y_test = load_images(df_test)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)
```
▶️ **Ejecutar**: Carga las imágenes, las redimensiona a 128x128 y las normaliza

---

#### 🤖 **Celda 7: Definición y Entrenamiento de Modelos**
```python
import tensorflow as tf
from tensorflow.keras import layers, models, applications

input_shape = (128, 128, 3)
num_classes = 3
EPOCHS = 30
BATCH_SIZE = 8

# Modelo 1: CNN sencilla
def build_cnn_simple(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo 2: CNN más profunda
def build_cnn_deep(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo 3: MobileNetV2 Transfer Learning
def build_mobilenet(input_shape, num_classes):
    base = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False  # Congelar base
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo 4: ResNet50 Transfer Learning (Nuevo)
def build_resnet50(input_shape, num_classes):
    base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False # Congelar base
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("Entrenando CNN Simple...")
cnn_simple = build_cnn_simple(input_shape, num_classes)
history1 = cnn_simple.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    verbose=2
)

print("\nEntrenando CNN Profunda...")
cnn_deep = build_cnn_deep(input_shape, num_classes)
history2 = cnn_deep.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    verbose=2
)

print("\nEntrenando MobileNetV2...")
mobilenet = build_mobilenet(input_shape, num_classes)
history3 = mobilenet.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    verbose=2
)

print("\nEntrenando ResNet50...")
resnet50 = build_resnet50(input_shape, num_classes)
history4 = resnet50.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    verbose=2
)

# Guardar modelos
cnn_simple.save('cnn_simple.h5')
cnn_deep.save('cnn_deep.h5')
mobilenet.save('mobilenet.h5')
resnet50.save('resnet50.h5') # Guardar el nuevo modelo
```
▶️ **Ejecutar**: 
- ⚠️ **IMPORTANTE**: Esta celda toma aproximadamente 4-5 horas en ejecutarse
- 🏗️ Define y entrena las 4 arquitecturas CNN
- 💾 Guarda los modelos entrenados

---

#### 📱 **Celda 8: Instalación de Dependencias para Streamlit**
```python
!pip install streamlit tensorflow pillow scikit-learn matplotlib
```
▶️ **Ejecutar**: Instala las dependencias necesarias para la interfaz web

---

#### 📄 **Celda 9: Configuración para Generación de PDF**
```python
!apt-get install -y wkhtmltopdf
!pip install pdfkit
```
▶️ **Ejecutar**: Instala herramientas para generar reportes en PDF

---

#### 🖥️ **Celda 10: Creación de la Aplicación Web Multiidioma**
```python
%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from io import BytesIO
import base64
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import cv2
import requests
import io

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

LANGUAGES = {
    'es': {'name': 'Español', 'flag': '🇪🇸'},
    'en': {'name': 'English', 'flag': '🇺🇸'},
    'it': {'name': 'Italiano', 'flag': '🇮🇹'}
}

TRANSLATIONS = {
    'es': {
        # Títulos principales
        'page_title': 'Diagnóstico Inteligente de Cacao - Deep Learning',
        'main_title': 'Diagnóstico Inteligente de Enfermedades del Cacao',
        'subtitle': 'Sistema avanzado con modelos deep learning (CNN Simple, CNN Profunda, MobileNetV2, ResNet50/GradCAM)',
        
        # Menús
        'control_panel': 'Panel de Control',
        'select_mode': 'Seleccionar Modo',
        'individual_diagnosis': 'Diagnóstico Individual',
        'disease_guide': 'Guía de Enfermedades',
        'comparative_analysis': 'Análisis Comparativo',
        'system_info': 'Información del Sistema',
        
        # Diagnóstico individual
        'select_image_source': 'Selecciona el origen de la imagen',
        'upload_from_computer': 'Subir desde mi computadora',
        'select_from_github': 'Seleccionar desde GitHub',
        'upload_image': 'Suba una imagen de mazorca de cacao para diagnóstico',
        'searching_images': 'Buscando imágenes públicas en la raíz del repositorio GitHub...',
        'choose_image': 'Elige una imagen',
        'load_selected_image': 'Cargar imagen seleccionada',
        'image_loaded': 'Imagen cargada',
        'automatic_diagnosis': 'Diagnóstico automático con IA',
        'processing': 'Procesando...',
        'ai_diagnosis': 'Diagnóstico por IA',
        'description': 'Descripción',
        'severity': 'Severidad',
        'gradcam_caption': 'Mapa de calor Grad-CAM (ResNet50)',
        'gradcam_description': 'Grad-CAM: Muestra qué regiones de la imagen influyeron más en la predicción del modelo ResNet50.',
        'model_confidence': 'Confianza de los modelos',
        'treatment_prevention': 'Tratamiento y medidas preventivas',
        'recommended_treatment': 'Tratamiento recomendado',
        'prevention': 'Prevención',
        'image_features': 'Características de la imagen cargada',
        'download_report': 'Descargar reporte individual',
        'download_html': 'Descargar reporte HTML',
        'download_pdf': 'Descargar reporte PDF',
        
        # Características de imagen
        'avg_brightness': 'Brillo promedio',
        'texture_variance': 'Varianza de textura',
        'contrast': 'Contraste',
        
        # Guía de enfermedades
        'disease_guide_title': 'Guía Completa de Enfermedades del Cacao',
        'example_image': 'Ejemplo de',
        'example_not_available': 'Imagen de ejemplo no disponible para',
        'general_info': 'Información General',
        'symptoms': 'Síntomas Característicos',
        'preventive_measures': 'Medidas Preventivas',
        'recommended_treatment': 'Tratamiento Recomendado',
        
        # Análisis comparativo
        'comparative_analysis_title': 'Análisis Comparativo de Modelos',
        'performance_analysis': 'Análisis de Rendimiento',
        'generate_report': 'Generar Reporte',
        'model_evaluation': 'Evaluación de Modelos de IA',
        'instructions': 'Instrucciones',
        'download_example': 'Descargar archivo de ejemplo',
        'select_csv_source': 'Selecciona el origen del CSV',
        'load_csv': 'Cargar archivo de resultados CSV',
        'searching_csv': 'Buscando archivos CSV públicos en la raíz del repositorio GitHub...',
        'choose_csv': 'Elige un archivo CSV',
        'load_selected_csv': 'Cargar CSV seleccionado',
        'file_loaded': 'Archivo cargado exitosamente',
        'samples_found': 'muestras encontradas',
        'best_model': 'El mejor modelo es',
        'statistical_analysis': 'Análisis Estadístico (Prueba de McNemar)',
        'significant': 'Significativa',
        'not_significant': 'No significativa',
        'generate_comparative_report': 'Generar Reporte Comparativo',
        'generating_report': 'Generando reporte completo...',
        'report_generated': 'Reporte PDF generado exitosamente',
        'pdfkit_not_available': 'pdfkit no disponible. Descargando como HTML.',
        
        # Métricas
        'precision': 'Precisión',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'mcc': 'MCC',
        'avg_specificity': 'Especificidad Promedio',
        'model': 'Modelo',
        'confidence': 'Confianza',
        'prediction': 'Predicción',
        
        # Enfermedades
        'healthy': 'Sano',
        'black_pod_rot': 'Pudrición Negra (Black Pod Rot)',
        'pod_borer': 'Barrenador de Mazorca (Pod Borer)',
        
        # Mensajes
        'error_loading_models': 'Error cargando modelos',
        'error_loading_image': 'No se pudo cargar la imagen seleccionada desde GitHub',
        'error_loading_csv': 'No se pudo cargar el archivo CSV seleccionado desde GitHub',
        'no_images_found': 'No se encontraron imágenes en la raíz del repositorio',
        'no_csv_found': 'No se encontraron archivos CSV en la raíz del repositorio',
        'csv_loaded_success': 'CSV cargado correctamente desde GitHub',
        'image_loaded_success': 'Imagen cargada correctamente desde GitHub',
        'analysis_first': 'Primero realiza el análisis en la pestaña Análisis de Rendimiento',
        'report_includes': 'El reporte incluirá métricas detalladas, matrices de confusión y análisis estadístico.',
        
        # Información del sistema
        'available_models': 'Modelos Disponibles',
        'detected_diseases': 'Enfermedades Detectadas',
        'healthy_pods': 'Mazorcas Sanas',
        'black_rot': 'Pudrición Negra',
        'pod_borer_pest': 'Barrenador de Mazorca',
        
        # Severidad
        'none': 'Ninguna',
        'high': 'Alta',
        'medium_high': 'Media-Alta',
        
        # Instrucciones detalladas
        'detailed_instructions': [
            'Sube un archivo CSV con las columnas: y_true, y_pred1, y_pred2, y_pred3',
            'El sistema comparará automáticamente los tres modelos',
            'Genera métricas avanzadas y pruebas estadísticas'
        ]
    },
    'en': {
        # Main titles
        'page_title': 'Intelligent Cacao Diagnosis - Deep Learning',
        'main_title': 'Intelligent Cacao Disease Diagnosis',
        'subtitle': 'Advanced system with deep learning models (Simple CNN, Deep CNN, MobileNetV2, ResNet50/GradCAM)',
        
        # Menus
        'control_panel': 'Control Panel',
        'select_mode': 'Select Mode',
        'individual_diagnosis': 'Individual Diagnosis',
        'disease_guide': 'Disease Guide',
        'comparative_analysis': 'Comparative Analysis',
        'system_info': 'System Information',
        
        # Individual diagnosis
        'select_image_source': 'Select image source',
        'upload_from_computer': 'Upload from my computer',
        'select_from_github': 'Select from GitHub',
        'upload_image': 'Upload a cacao pod image for diagnosis',
        'searching_images': 'Searching for public images in the GitHub repository root...',
        'choose_image': 'Choose an image',
        'load_selected_image': 'Load selected image',
        'image_loaded': 'Image loaded',
        'automatic_diagnosis': 'Automatic AI diagnosis',
        'processing': 'Processing...',
        'ai_diagnosis': 'AI Diagnosis',
        'description': 'Description',
        'severity': 'Severity',
        'gradcam_caption': 'Grad-CAM heatmap (ResNet50)',
        'gradcam_description': 'Grad-CAM: Shows which regions of the image most influenced the ResNet50 model prediction.',
        'model_confidence': 'Model confidence',
        'treatment_prevention': 'Treatment and preventive measures',
        'recommended_treatment': 'Recommended treatment',
        'prevention': 'Prevention',
        'image_features': 'Loaded image features',
        'download_report': 'Download individual report',
        'download_html': 'Download HTML report',
        'download_pdf': 'Download PDF report',
        
        # Image features
        'avg_brightness': 'Average brightness',
        'texture_variance': 'Texture variance',
        'contrast': 'Contrast',
        
        # Disease guide
        'disease_guide_title': 'Complete Cacao Disease Guide',
        'example_image': 'Example of',
        'example_not_available': 'Example image not available for',
        'general_info': 'General Information',
        'symptoms': 'Characteristic Symptoms',
        'preventive_measures': 'Preventive Measures',
        'recommended_treatment': 'Recommended Treatment',
        
        # Comparative analysis
        'comparative_analysis_title': 'Model Comparative Analysis',
        'performance_analysis': 'Performance Analysis',
        'generate_report': 'Generate Report',
        'model_evaluation': 'AI Model Evaluation',
        'instructions': 'Instructions',
        'download_example': 'Download example file',
        'select_csv_source': 'Select CSV source',
        'load_csv': 'Load CSV results file',
        'searching_csv': 'Searching for public CSV files in the GitHub repository root...',
        'choose_csv': 'Choose a CSV file',
        'load_selected_csv': 'Load selected CSV',
        'file_loaded': 'File loaded successfully',
        'samples_found': 'samples found',
        'best_model': 'The best model is',
        'statistical_analysis': 'Statistical Analysis (McNemar Test)',
        'significant': 'Significant',
        'not_significant': 'Not significant',
        'generate_comparative_report': 'Generate Comparative Report',
        'generating_report': 'Generating complete report...',
        'report_generated': 'PDF report generated successfully',
        'pdfkit_not_available': 'pdfkit not available. Downloading as HTML.',
        
        # Metrics
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'mcc': 'MCC',
        'avg_specificity': 'Average Specificity',
        'model': 'Model',
        'confidence': 'Confidence',
        'prediction': 'Prediction',
        
        # Diseases
        'healthy': 'Healthy',
        'black_pod_rot': 'Black Pod Rot',
        'pod_borer': 'Pod Borer',
        
        # Messages
        'error_loading_models': 'Error loading models',
        'error_loading_image': 'Could not load selected image from GitHub',
        'error_loading_csv': 'Could not load selected CSV file from GitHub',
        'no_images_found': 'No images found in repository root',
        'no_csv_found': 'No CSV files found in repository root',
        'csv_loaded_success': 'CSV loaded successfully from GitHub',
        'image_loaded_success': 'Image loaded successfully from GitHub',
        'analysis_first': 'First perform the analysis in the Performance Analysis tab',
        'report_includes': 'The report will include detailed metrics, confusion matrices and statistical analysis.',
        
        # System information
        'available_models': 'Available Models',
        'detected_diseases': 'Detected Diseases',
        'healthy_pods': 'Healthy Pods',
        'black_rot': 'Black Rot',
        'pod_borer_pest': 'Pod Borer',
        
        # Severity
        'none': 'None',
        'high': 'High',
        'medium_high': 'Medium-High',
        
        # Detailed instructions
        'detailed_instructions': [
            'Upload a CSV file with columns: y_true, y_pred1, y_pred2, y_pred3',
            'The system will automatically compare the three models',
            'Generate advanced metrics and statistical tests'
        ]
    },
    'it': {
        # Titoli principali
        'page_title': 'Diagnosi Intelligente del Cacao - Deep Learning',
        'main_title': 'Diagnosi Intelligente delle Malattie del Cacao',
        'subtitle': 'Sistema avanzato con modelli deep learning (CNN Semplice, CNN Profonda, MobileNetV2, ResNet50/GradCAM)',
        
        # Menu
        'control_panel': 'Pannello di Controllo',
        'select_mode': 'Seleziona Modalità',
        'individual_diagnosis': 'Diagnosi Individuale',
        'disease_guide': 'Guida alle Malattie',
        'comparative_analysis': 'Analisi Comparativa',
        'system_info': 'Informazioni del Sistema',
        
        # Diagnosi individuale
        'select_image_source': 'Seleziona la fonte dell\'immagine',
        'upload_from_computer': 'Carica dal mio computer',
        'select_from_github': 'Seleziona da GitHub',
        'upload_image': 'Carica un\'immagine di baccello di cacao per la diagnosi',
        'searching_images': 'Ricerca immagini pubbliche nella radice del repository GitHub...',
        'choose_image': 'Scegli un\'immagine',
        'load_selected_image': 'Carica immagine selezionata',
        'image_loaded': 'Immagine caricata',
        'automatic_diagnosis': 'Diagnosi automatica con IA',
        'processing': 'Elaborazione...',
        'ai_diagnosis': 'Diagnosi IA',
        'description': 'Descrizione',
        'severity': 'Severità',
        'gradcam_caption': 'Mappa di calore Grad-CAM (ResNet50)',
        'gradcam_description': 'Grad-CAM: Mostra quali regioni dell\'immagine hanno influenzato maggiormente la predizione del modello ResNet50.',
        'model_confidence': 'Confidenza dei modelli',
        'treatment_prevention': 'Trattamento e misure preventive',
        'recommended_treatment': 'Trattamento raccomandato',
        'prevention': 'Prevenzione',
        'image_features': 'Caratteristiche dell\'immagine caricata',
        'download_report': 'Scarica rapporto individuale',
        'download_html': 'Scarica rapporto HTML',
        'download_pdf': 'Scarica rapporto PDF',
        
        # Caratteristiche immagine
        'avg_brightness': 'Luminosità media',
        'texture_variance': 'Varianza texture',
        'contrast': 'Contrasto',
        
        # Guida malattie
        'disease_guide_title': 'Guida Completa alle Malattie del Cacao',
        'example_image': 'Esempio di',
        'example_not_available': 'Immagine di esempio non disponibile per',
        'general_info': 'Informazioni Generali',
        'symptoms': 'Sintomi Caratteristici',
        'preventive_measures': 'Misure Preventive',
        'recommended_treatment': 'Trattamento Raccomandato',
        
        # Analisi comparativa
        'comparative_analysis_title': 'Analisi Comparativa dei Modelli',
        'performance_analysis': 'Analisi delle Prestazioni',
        'generate_report': 'Genera Rapporto',
        'model_evaluation': 'Valutazione Modelli IA',
        'instructions': 'Istruzioni',
        'download_example': 'Scarica file di esempio',
        'select_csv_source': 'Seleziona la fonte del CSV',
        'load_csv': 'Carica file CSV dei risultati',
        'searching_csv': 'Ricerca file CSV pubblici nella radice del repository GitHub...',
        'choose_csv': 'Scegli un file CSV',
        'load_selected_csv': 'Carica CSV selezionato',
        'file_loaded': 'File caricato con successo',
        'samples_found': 'campioni trovati',
        'best_model': 'Il modello migliore è',
        'statistical_analysis': 'Analisi Statistica (Test di McNemar)',
        'significant': 'Significativa',
        'not_significant': 'Non significativa',
        'generate_comparative_report': 'Genera Rapporto Comparativo',
        'generating_report': 'Generazione rapporto completo...',
        'report_generated': 'Rapporto PDF generato con successo',
        'pdfkit_not_available': 'pdfkit non disponibile. Scaricando come HTML.',
        
        # Metriche
        'precision': 'Precisione',
        'recall': 'Richiamo',
        'f1_score': 'F1-Score',
        'mcc': 'MCC',
        'avg_specificity': 'Specificità Media',
        'model': 'Modello',
        'confidence': 'Confidenza',
        'prediction': 'Predizione',
        
        # Malattie
        'healthy': 'Sano',
        'black_pod_rot': 'Marciume Nero del Baccello',
        'pod_borer': 'Perforatore del Baccello',
        
        # Messaggi
        'error_loading_models': 'Errore nel caricamento dei modelli',
        'error_loading_image': 'Impossibile caricare l\'immagine selezionata da GitHub',
        'error_loading_csv': 'Impossibile caricare il file CSV selezionato da GitHub',
        'no_images_found': 'Nessuna immagine trovata nella radice del repository',
        'no_csv_found': 'Nessun file CSV trovato nella radice del repository',
        'csv_loaded_success': 'CSV caricato con successo da GitHub',
        'image_loaded_success': 'Immagine caricata con successo da GitHub',
        'analysis_first': 'Prima esegui l\'analisi nella scheda Analisi delle Prestazioni',
        'report_includes': 'Il rapporto includerà metriche dettagliate, matrici di confusione e analisi statistica.',
        
        # Informazioni sistema
        'available_models': 'Modelli Disponibili',
        'detected_diseases': 'Malattie Rilevate',
        'healthy_pods': 'Baccelli Sani',
        'black_rot': 'Marciume Nero',
        'pod_borer_pest': 'Perforatore del Baccello',
        
        # Severità
        'none': 'Nessuna',
        'high': 'Alta',
        'medium_high': 'Media-Alta',
        
        # Istruzioni dettagliate
        'detailed_instructions': [
            'Carica un file CSV con le colonne: y_true, y_pred1, y_pred2, y_pred3',
            'Il sistema confronterà automaticamente i tre modelli',
            'Genera metriche avanzate e test statistici'
        ]
    }
}

def get_text(key, lang='es'):
    return TRANSLATIONS.get(lang, TRANSLATIONS['es']).get(key, key)

def init_language():
    if 'language' not in st.session_state:
        st.session_state.language = 'es'

def language_selector():
    st.sidebar.markdown("### 🌐 Idioma / Language / Lingua")
    language_options = [f"{LANGUAGES[lang]['flag']} {LANGUAGES[lang]['name']}" for lang in LANGUAGES.keys()]
    current_lang_index = list(LANGUAGES.keys()).index(st.session_state.language)
    selected_lang = st.sidebar.selectbox(
        "Seleccionar idioma",
        language_options,
        index=current_lang_index,
        key="language_selector"
    )
    selected_lang_code = list(LANGUAGES.keys())[language_options.index(selected_lang)]
    if selected_lang_code != st.session_state.language:
        st.session_state.language = selected_lang_code
        st.rerun()

GITHUB_USER = "DanteVas"
GITHUB_REPO = "Sistema-de-diagnostico-de-enfermedades-del-cacao-usando-deep-learning"
GITHUB_IMG_FOLDER = ""

def listar_imagenes_github(user, repo, carpeta):
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{carpeta}".rstrip("/")
    r = requests.get(api_url)
    files = []
    if r.status_code == 200:
        content = r.json()
        for file in content:
            if file['type'] == 'file' and (file['name'].lower().endswith('.jpg') or file['name'].lower().endswith('.png')):
                files.append(file['name'])
    return files

def listar_archivos_github(user, repo, carpeta, extensiones):
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{carpeta}".rstrip("/")
    r = requests.get(api_url)
    files = []
    if r.status_code == 200:
        content = r.json()
        for file in content:
            if file['type'] == 'file' and any(file['name'].lower().endswith(ext) for ext in extensiones):
                files.append(file['name'])
    return files

def get_github_image_url(user, repo, carpeta, filename):
    if carpeta == "":
        return f"https://raw.githubusercontent.com/{user}/{repo}/main/{filename}"
    else:
        return f"https://raw.githubusercontent.com/{user}/{repo}/main/{carpeta}/{filename}"

def cargar_imagen_desde_github(url):
    r = requests.get(url)
    if r.status_code == 200:
        return Image.open(BytesIO(r.content)).convert("RGB")
    else:
        return None

DISEASE_INFO = {
    0: {'name': 'Sano', 'desc': 'Mazorca sin síntomas evidentes. Color uniforme y sin lesiones visibles.',
        'treatment': 'No requiere tratamiento. Mantenga prácticas agrícolas preventivas.',
        'symptoms': 'Fruto y hoja sin manchas ni daños.',
        'prevention': 'Mantener buen drenaje, inspección regular, manejo integrado de plagas.',
        'severity': 'Ninguna', 'color': '#43A047', 'bg': '#E9FCE7', 'class': 'healthy'},
    1: {'name': 'Pudrición Negra (Black Pod Rot)', 'desc': 'Enfermedad causada por Phytophthora spp., con manchas marrón-negruzcas, generalmente en la base o extremo de la mazorca.',
        'treatment': '1. Retirar y destruir frutos enfermos\n2. Aplicar fungicidas (a base de cobre)\n3. Mejorar el drenaje del suelo\n4. Podar ramas bajas\n5. Aplicar materia orgánica',
        'symptoms': 'Manchas marrón oscuro o negras, consistencia húmeda, progresión rápida.',
        'prevention': 'Mejorar ventilación, evitar humedad excesiva, aplicar fungicidas preventivos.',
        'severity': 'Alta', 'color': '#6D4C41', 'bg': '#F9ECE4', 'class': 'pudricion'},
    2: {'name': 'Barrenador de Mazorca (Pod Borer)', 'desc': 'Plaga causada por larvas que se alimentan del interior de la mazorca. Provoca daños internos, galerías y pérdida de semillas.',
        'treatment': '1. Recolectar y destruir frutos infestados\n2. Uso de trampas de feromonas\n3. Control biológico\n4. Aplicaciones de insecticidas específicos\n5. Limpieza del cultivo',
        'symptoms': 'Orificios pequeños, daños en semillas, frutos deformados, presencia de larvas.',
        'prevention': 'Monitoreo con trampas, control biológico, manejo de residuos.',
        'severity': 'Media-Alta', 'color': '#FFB300', 'bg': '#FFF8E1', 'class': 'barrenador'}
}
CLASSES = [DISEASE_INFO[i]['name'] for i in range(3)]
MODEL_NAMES = ['CNN Simple', 'CNN Profunda', 'MobileNetV2']
IMG_HEIGHT_GRADCAM, IMG_WIDTH_GRADCAM = 224, 224

METRIC_DESCRIPTIONS = {
    "Precisión": "Porcentaje de predicciones correctas sobre el total de muestras.",
    "Recall": "Capacidad del modelo para encontrar todos los casos positivos (sensibilidad).",
    "F1-Score": "Promedio armónico entre precisión y recall; balance entre falsos positivos y negativos.",
    "MCC": "Medida global de calidad del modelo (-1 a 1, donde 1 es perfecto, 0 es azar).",
    "Especificidad Promedio": "Capacidad de identificar correctamente los negativos (verdaderos negativos / (verdaderos negativos + falsos positivos))."
}

def plot_confusion_matrix_mpl(y_true, y_pred, classes, title='Matriz de Confusión'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

def create_confidence_chart(predictions, model_names):
    df_conf = pd.DataFrame({
        'Modelo': model_names,
        'Confianza': [float(np.max(pred)) * 100 for pred in predictions],
        'Predicción': [CLASSES[int(np.argmax(pred))] for pred in predictions]
    })
    fig = px.bar(df_conf, x='Modelo', y='Confianza',
                 color='Predicción',
                 title='Confianza por Modelo',
                 color_discrete_map={
                     'Sano': '#43A047',
                     'Pudrición Negra (Black Pod Rot)': '#6D4C41',
                     'Barrenador de Mazorca (Pod Borer)': '#FFB300'
                 })
    fig.update_layout(height=400)
    return fig

def mcc_per_class(y_true, y_pred, n_classes=3):
    mccs = []
    for c in range(n_classes):
        y_true_bin = (np.array(y_true) == c).astype(int)
        y_pred_bin = (np.array(y_pred) == c).astype(int)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        mccs.append(mcc)
    return mccs

def calculate_advanced_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(len(CLASSES)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    return {
        'report': report,
        'specificity': specificity,
        'accuracy': accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'mcc_per_class': mcc_per_class(y_true, y_pred)
    }

def mcnemar_test(y_true, y_pred1, y_pred2):
    table = np.zeros((2, 2))
    for t, p1, p2 in zip(y_true, y_pred1, y_pred2):
        if p1 == t and p2 != t:
            table[0, 1] += 1
        elif p1 != t and p2 == t:
            table[1, 0] += 1
    try:
        result = mcnemar(table, exact=True)
        return result.statistic, result.pvalue
    except:
        return 0, 1

def preprocess_image(image, target_size=(128, 128)):
    img = np.array(image)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    try:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, target_size)
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception:
        img_processed = Image.fromarray(img).resize(target_size)
        img_processed = np.array(img_processed)
    img_processed = img_processed / 255.0
    return np.expand_dims(img_processed, axis=0)

def analyze_image_features(image):
    img_array = np.array(image)
    avg_color = np.mean(img_array, axis=(0, 1))
    brightness = np.mean(avg_color)
    gray = np.mean(img_array, axis=2)
    texture_variance = np.var(gray)
    contrast = np.std(gray)
    return {
        'brightness': brightness,
        'texture_variance': texture_variance,
        'contrast': contrast,
        'avg_color': avg_color.tolist()
    }

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_array = tf.cast(img_array, tf.float32)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        heatmap = heatmap
    else:
        heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()

def superimpose_heatmap(original_img_pil, heatmap):
    img_display = np.array(original_img_pil.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    colormap = plt.cm.jet
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    alpha = 0.4
    superimposed_img = cv2.addWeighted(img_display, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(superimposed_img)

@st.cache_resource
def load_models():
    try:
        model1 = load_model('cnn_simple.h5')
        model2 = load_model('cnn_deep.h5')
        model3 = load_model('mobilenet.h5')
        base_model_resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT_GRADCAM, IMG_WIDTH_GRADCAM, 3))
        base_model_resnet.trainable = False
        model_resnet_for_gradcam = tf.keras.Sequential([
            base_model_resnet,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
        model_resnet_for_gradcam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model1, model2, model3, model_resnet_for_gradcam, base_model_resnet
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None, None, None

def generate_html_download(html_content, filename_prefix="reporte", lang="es"):
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename_prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">{get_text("download_html", lang)}</a>'
    st.markdown(href, unsafe_allow_html=True)

def generate_pdf_report(html_content, filename_prefix="reporte", lang="es"):
    if PDFKIT_AVAILABLE:
        tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp_html.write(html_content.encode('utf-8'))
        tmp_html.close()
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdfkit.from_file(tmp_html.name, pdf_file.name)
        pdf_file.close()
        with open(pdf_file.name, "rb") as f:
            st.download_button(
                label=f"📄 {get_text('download_pdf', lang)}",
                data=f,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        os.unlink(tmp_html.name)
        os.unlink(pdf_file.name)
    else:
        st.info(get_text("pdfkit_not_available", lang))

def generate_individual_report(image, predictions, features, consensus, lang):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width: 300px; border-radius: 12px; margin: 16px auto; display: block;">'
    gradcam_html = ""
    if 'gradcam_img_base64' in st.session_state:
        gradcam_html = f"""
        <div class="section">
            <h2>{get_text("gradcam_caption", lang)}</h2>
            <img src="data:image/png;base64,{st.session_state.gradcam_img_base64}" style="max-width: 500px; border-radius: 12px; margin: 16px auto; display: block;">
            <p style="text-align: center; font-style: italic;">{get_text("gradcam_description", lang)}</p>
        </div>
        """
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>{get_text('download_report', lang)} - Cacao</title>
    <style>
    body {{
        font-family: 'Segoe UI', Arial, sans-serif;
        margin: 32px;
        background: #FAF6F0;
        color: #333;
    }}
    .header {{
        text-align: center;
        margin-bottom: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
    }}
    .main-title {{
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
    }}
    .subtitle {{
        font-size: 1.2em;
        margin: 10px 0 0 0;
        opacity: 0.9;
    }}
    .section {{
        background: white;
        padding: 25px;
        margin: 20px 0;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    .diagnosis-card {{
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }}
    th, td {{
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
    th {{
        background: #B9F6CA;
        color: #2E7D32;
        font-weight: 600;
    }}
    .footer {{
        text-align: center;
        margin-top: 40px;
        color: #666;
        font-size: 0.9em;
    }}
    .feature-box {{
        display: inline-block;
        margin: 10px;
        padding: 15px;
        background: #f0f8ff;
        border-radius: 10px;
        text-align: center;
    }}
    </style>
    </head>
    <body>
    <div class="header">
    <h1 class="main-title"> 🍫  {get_text('download_report', lang)}</h1>
    <p class="subtitle">{get_text('report_generated', lang)} {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
    </div>
    {img_html}
    """
    disease = DISEASE_INFO[consensus]
    html += f"""
    <div class="diagnosis-card">
    <h2>{get_text('ai_diagnosis', lang)}: {disease['name']}</h2>
    <p style="font-size: 1.2em; margin: 10px 0;">{disease['desc']}</p>
    <p><strong>{get_text('severity', lang)}:</strong> {disease['severity']}</p>
    </div>
    """
    html += f"""
    <div class="section">
    <h2>{get_text('model_confidence', lang)}</h2>
    <div style="text-align: center;">
    """
    for i, (pred, name) in enumerate(zip(predictions, MODEL_NAMES)):
        idx = int(np.argmax(pred))
        conf = float(np.max(pred)) * 100
        pred_disease = DISEASE_INFO[idx]
        html += f"""
        <div class="model-result">
        <h3>{name}</h3>
        <p><strong>{get_text('prediction', lang)}:</strong> {pred_disease['name']}</p>
        <p><strong>{get_text('confidence', lang)}:</strong> {conf:.1f}%</p>
        <div class="confidence-bar">
        <div class="confidence-fill" style="width: {conf}%;"></div>
        </div>
        </div>
        """
    html += """
    </div>
    </div>
    """
    html += f"""
    <div class="section">
    <h2>{get_text('image_features', lang)}</h2>
    <div style="text-align: center;">
    <div class="feature-box">
    <h4>{get_text('avg_brightness', lang)}</h4>
    <p>{features['brightness']:.2f}</p>
    </div>
    <div class="feature-box">
    <h4>{get_text('texture_variance', lang)}</h4>
    <p>{features['texture_variance']:.2f}</p>
    </div>
    <div class="feature-box">
    <h4>{get_text('contrast', lang)}</h4>
    <p>{features['contrast']:.2f}</p>
    </div>
    </div>
    </div>
    """
    html += gradcam_html
    html += f"""
    <div class="section">
    <h2>{get_text('recommended_treatment', lang)}</h2>
    <pre style="white-space: pre-wrap; font-family: inherit;">{disease['treatment']}</pre>
    <h3>{get_text('prevention', lang)}</h3>
    <p>{disease['prevention']}</p>
    </div>
    """
    html += f"""
    <div class="section">
    <h2>{get_text('model_confidence', lang)}</h2>
    <table>
    <tr><th>{get_text('model', lang)}</th><th>{get_text('confidence', lang)}</th></tr>
    """
    avg_probs = np.mean(predictions, axis=0)
    for i, (class_name, prob) in enumerate(zip(CLASSES, avg_probs)):
        html += f"<tr><td>{class_name}</td><td>{prob*100:.2f}%</td></tr>"
    html += """
    </table>
    </div>
    """
    html += f"""
    <div class="footer">
    <p>Desarrollado por Galdos Hilda y Dante Vásquez © {datetime.now().year}</p>
    <p>Sistema de Diagnóstico Inteligente para Enfermedades del Cacao</p>
    </div>
    </body>
    </html>
    """
    return html

def interpret_mcc(mcc, lang):
    if mcc > 0.8:
        return get_text("mcc_interpret_excellent", lang) if "mcc_interpret_excellent" in TRANSLATIONS[lang] else "Excelente capacidad de discriminación (MCC alto)"
    elif mcc > 0.6:
        return get_text("mcc_interpret_good", lang) if "mcc_interpret_good" in TRANSLATIONS[lang] else "Buena capacidad de discriminación"
    elif mcc > 0.4:
        return get_text("mcc_interpret_moderate", lang) if "mcc_interpret_moderate" in TRANSLATIONS[lang] else "Moderada capacidad de discriminación"
    elif mcc > 0.2:
        return get_text("mcc_interpret_weak", lang) if "mcc_interpret_weak" in TRANSLATIONS[lang] else "Débil capacidad de discriminación"
    else:
        return get_text("mcc_interpret_poor", lang) if "mcc_interpret_poor" in TRANSLATIONS[lang] else "Pobre capacidad de discriminación (similar al azar)"

def generate_comparative_report(metrics, confusion_images, model_names, y_true, predictions, lang):
    mccs = [m['mcc'] for m in metrics[:len(model_names)]]
    best_idx = int(np.argmax(mccs))
    best_model = model_names[best_idx]
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <title>{get_text('comparative_analysis_title', lang)}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; background: #FAF6F0; color: #333; }}
        h1 {{ color: #4E342E; font-size: 2.2em; }}
        h2 {{ color: #388e3c; margin-top: 36px; }}
        table, th, td {{ border: 1px solid #999; border-collapse: collapse; }}
        th, td {{ padding: 9px 14px; }}
        th {{ background: #B9F6CA; color: #2E7D32; font-weight: 600; }}
        .img-box {{ display:inline-block; margin:10px; border-radius:8px; box-shadow:0 4px 12px #0001; background:#fff; }}
        .footer {{font-size: 0.95em; color: #888; margin-top: 2em; text-align: center;}}
        .card-title {{ font-size: 1.2em; font-weight: 700; margin-top: 14px; color: #4E342E; }}
        .desc-block {{ background: #f7fbe7; border-radius: 8px; padding: 12px 18px; margin-bottom: 15px; border-left: 5px solid #7bc043; }}
    </style>
    </head>
    <body>
        <h1>{get_text('comparative_analysis_title', lang)}</h1>
        <p>{get_text('report_generated', lang)} {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
        <div class="desc-block">
            <h3>{get_text('performance_analysis', lang)}</h3>
            <ul>
                <li><b>{get_text('precision', lang)}:</b> {METRIC_DESCRIPTIONS['Precisión']}</li>
                <li><b>{get_text('recall', lang)}:</b> {METRIC_DESCRIPTIONS['Recall']}</li>
                <li><b>{get_text('f1_score', lang)}:</b> {METRIC_DESCRIPTIONS['F1-Score']}</li>
                <li><b>{get_text('mcc', lang)}:</b> {METRIC_DESCRIPTIONS['MCC']}</li>
                <li><b>{get_text('avg_specificity', lang)}:</b> {METRIC_DESCRIPTIONS['Especificidad Promedio']}</li>
            </ul>
        </div>
        <h2>{get_text('performance_analysis', lang)}</h2>
        <table>
            <tr>
                <th>{get_text('model', lang)}</th>
                <th>{get_text('precision', lang)}</th>
                <th>{get_text('recall', lang)}</th>
                <th>{get_text('f1_score', lang)}</th>
                <th>{get_text('mcc', lang)}</th>
                <th>Interpretación MCC</th>
                <th>{get_text('avg_specificity', lang)}</th>
            </tr>
    """
    n_models = min(len(model_names), len(metrics))
    for i in range(n_models):
        html += f"<tr><td>{model_names[i]}</td>"
        html += f"<td>{metrics[i]['accuracy']:.3f}</td>"
        html += f"<td>{metrics[i]['report']['macro avg']['recall']:.3f}</td>"
        html += f"<td>{metrics[i]['report']['macro avg']['f1-score']:.3f}</td>"
        html += f"<td>{metrics[i]['mcc']:.3f}</td>"
        html += f"<td>{interpret_mcc(metrics[i]['mcc'], lang)}</td>"
        html += f"<td>{np.mean(metrics[i]['specificity']):.3f}</td></tr>"
    html += "</table>"

    html += f"<p><b>{get_text('best_model', lang)}:</b> <b>{best_model}</b></p>"

    html += f"<h2>{get_text('mcc', lang)} por clase</h2>"
    html += f"<p>{get_text('mcc', lang)} por clase mide la capacidad del modelo para distinguir correctamente cada clase frente al resto.</p>"
    html += f"<table><tr><th>{get_text('model', lang)}</th>"
    for idx, name in enumerate(metrics[0]['report']):
        if name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        html += f"<th>{name}</th>"
    html += "</tr>"
    for i in range(n_models):
        html += f"<tr><td>{model_names[i]}</td>"
        for mcc_class in metrics[i]['mcc_per_class']:
            html += f"<td>{mcc_class:.3f}</td>"
        html += "</tr>"
    html += "</table>"

    html += f"<h2>{get_text('statistical_analysis', lang)}</h2><p>{get_text('report_includes', lang)}</p>"
    for i in range(n_models):
        html += f"<li><b>{model_names[i]} - {get_text('mcc', lang)}:</b> {metrics[i]['mcc']:.3f} &rarr; {interpret_mcc(metrics[i]['mcc'], lang)}.</li>"

    html += f"<br><b>{get_text('statistical_analysis', lang)}:</b> {get_text('best_model', lang)} ({best_model})"
    for i in range(n_models):
        if i == best_idx:
            continue
        stat, pval = mcnemar_test(y_true, predictions[best_idx], predictions[i])
        if pval < 0.05:
            conclusion = f"<b>{get_text('significant', lang)}</b>"
        else:
            conclusion = f"<b>{get_text('not_significant', lang)}</b>"
        html += f"<li>{best_model} vs {model_names[i]}: Estadístico = {stat:.3f}, p-valor = {pval:.4f}. {conclusion}</li>"
    html += """
    </ul>
    <div style='background:#f9fbe7; padding:10px; border-radius:8px;'>
        <b>Interpretación:</b> Un p-valor menor a 0.05 indica que existen diferencias estadísticas relevantes en el desempeño entre los modelos comparados.
    </div>
    """
    html += f"""
        <div class="footer">
            <p>Desarrollado por Galdos Hilda y Dante Vásquez © {datetime.now().year}</p>
            <p>Sistema de Diagnóstico Inteligente para Enfermedades del Cacao</p>
        </div>
    </body>
    </html>
    """
    return html

st.set_page_config(
    page_title="Diagnóstico Inteligente de Cacao - Deep Learning",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_language()
language_selector()
lang = st.session_state.language

st.sidebar.markdown("### 🎛️ Panel de Control")
app_mode = st.sidebar.selectbox(
    get_text("select_mode", lang),
    ["🔍 " + get_text('individual_diagnosis', lang), "📚 " + get_text('disease_guide', lang), "📊 " + get_text('comparative_analysis', lang)],
    help=get_text("select_mode", lang)
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 " + get_text("system_info", lang))
st.sidebar.info(f"""
**{get_text('available_models', lang)}:**
- CNN Simple
- CNN Profunda
- MobileNetV2

**{get_text('detected_diseases', lang)}:**
- {get_text('healthy_pods', lang)}
- {get_text('black_rot', lang)}
- {get_text('pod_borer_pest', lang)}
""")

model1, model2, model3, model_resnet, base_model_resnet = load_models()

if app_mode.startswith("🔍"):
    st.title("🍫 " + get_text('main_title', lang))
    st.markdown(get_text('subtitle', lang))

    origen_img = st.radio(get_text('select_image_source', lang), (get_text('upload_from_computer', lang), get_text('select_from_github', lang)))
    image = None

    if origen_img == get_text('upload_from_computer', lang):
        uploaded_file_img = st.file_uploader(get_text('upload_image', lang), type=["jpg", "jpeg", "png"], key="diagnostico_img")
        if uploaded_file_img:
            image = Image.open(uploaded_file_img).convert("RGB")
    elif origen_img == get_text('select_from_github', lang):
        st.markdown(get_text('searching_images', lang))
        imagenes_disponibles = listar_imagenes_github(GITHUB_USER, GITHUB_REPO, GITHUB_IMG_FOLDER)
        imagenes_disponibles = [f for f in imagenes_disponibles if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imagenes_disponibles:
            imagen_elegida = st.selectbox(get_text('choose_image', lang), imagenes_disponibles, key="img_github_select")
            if st.button(get_text('load_selected_image', lang)):
                url_img = get_github_image_url(GITHUB_USER, GITHUB_REPO, GITHUB_IMG_FOLDER, imagen_elegida)
                image_github = cargar_imagen_desde_github(url_img)
                if image_github:
                    image = image_github
                    st.success(f"{get_text('image_loaded_success', lang)} '{imagen_elegida}'.")
                else:
                    st.error(get_text('error_loading_image', lang))
        else:
            st.warning(get_text('no_images_found', lang))

    if image:
        st.image(image, caption=get_text('image_loaded', lang), use_container_width=True)
        st.markdown("---")
        st.markdown("## " + get_text('automatic_diagnosis', lang))
        with st.spinner(get_text('processing', lang)):
            img_pre = preprocess_image(image, (128, 128))
            pred1 = model1.predict(img_pre)[0]
            pred2 = model2.predict(img_pre)[0]
            pred3 = model3.predict(img_pre)[0]
            predictions = [pred1, pred2, pred3]

            consensus = int(np.round(np.mean([np.argmax(p) for p in predictions])))
            disease = DISEASE_INFO[consensus]

            resnet_input = preprocess_image(image, (IMG_HEIGHT_GRADCAM, IMG_WIDTH_GRADCAM))
            resnet_preds = model_resnet.predict(resnet_input)[0]
            resnet_pred_class = int(np.argmax(resnet_preds))
            heatmap = make_gradcam_heatmap(resnet_input, base_model_resnet, "conv5_block3_out", pred_index=resnet_pred_class)
            gradcam_img = superimpose_heatmap(image, heatmap)
            buffered_gradcam = BytesIO()
            gradcam_img.save(buffered_gradcam, format="PNG")
            st.session_state.gradcam_img_base64 = base64.b64encode(buffered_gradcam.getvalue()).decode()

            st.subheader(f"{get_text('ai_diagnosis', lang)}: **{disease['name']}**")
            st.markdown(f"**{get_text('description', lang)}:** {disease['desc']}")
            st.markdown(f"**{get_text('severity', lang)}:** {disease['severity']}")

            st.image(gradcam_img, caption=get_text('gradcam_caption', lang), use_container_width=False, width=420)
            st.markdown("> **" + get_text('gradcam_description', lang) + "**")

            st.markdown("### " + get_text('model_confidence', lang))
            fig_conf = create_confidence_chart(predictions, MODEL_NAMES)
            st.plotly_chart(fig_conf, use_container_width=True)

            st.markdown("### " + get_text('treatment_prevention', lang))
            st.markdown(f"**{get_text('recommended_treatment', lang)}:**\n{disease['treatment']}")
            st.markdown(f"**{get_text('prevention', lang)}:** {disease['prevention']}")

            features = analyze_image_features(image)
            st.markdown("### " + get_text('image_features', lang))
            st.write({
                get_text('avg_brightness', lang): f"{features['brightness']:.2f}",
                get_text('texture_variance', lang): f"{features['texture_variance']:.2f}",
                get_text('contrast', lang): f"{features['contrast']:.2f}"
            })

            html_report = generate_individual_report(image, predictions, features, consensus, lang)
            st.markdown("## 📄 " + get_text('download_report', lang))
            generate_html_download(html_report, lang=lang)
            if PDFKIT_AVAILABLE:
                generate_pdf_report(html_report, lang=lang)

elif app_mode.startswith("📚"):
    st.title("📚 " + get_text('disease_guide_title', lang))
    tabs = st.tabs([f"{DISEASE_INFO[i]['name']}" for i in range(3)])
    for i, tab in enumerate(tabs):
        with tab:
            disease = DISEASE_INFO[i]
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"### {disease['name']}")
                img_path = f"{i}.jpg"
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{get_text('example_image', lang)} {disease['name']}", use_container_width=True)
                else:
                    st.info(f"💡 {get_text('example_not_available', lang)} {disease['name']}")
                st.markdown(f"#### 📊 {get_text('general_info', lang)}")
                st.metric(get_text('severity', lang), disease['severity'])
            with col2:
                st.markdown(f"#### 📝 {get_text('description', lang)}")
                st.write(disease['desc'])
                st.markdown(f"#### 🔍 {get_text('symptoms', lang)}")
                st.write(disease['symptoms'])
                st.markdown(f"#### 🛡️ {get_text('preventive_measures', lang)}")
                st.write(disease['prevention'])
            st.markdown(f"#### 💊 {get_text('recommended_treatment', lang)}")
            st.markdown(f"""
            <div class="card {disease['class']}">
                <pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">{disease['treatment']}</pre>
            </div>
            """, unsafe_allow_html=True)

elif app_mode.startswith("📊"):
    st.title("📊 " + get_text('comparative_analysis_title', lang))
    tab1, tab2 = st.tabs(["🔬 " + get_text('performance_analysis', lang), "📄 " + get_text('generate_report', lang)])

    with tab1:
        st.markdown("### " + get_text('model_evaluation', lang))
        st.info("\n".join(["**" + get_text('instructions', lang) + ":**"] + [f"{i+1}. {s}" for i, s in enumerate(get_text('detailed_instructions', lang))]))

        with st.expander("📥 " + get_text('download_example', lang)):
            example_data = pd.DataFrame({
                "y_true": [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2] * 10,
                "y_pred1": [0,1,2,0,1,2,0,2,1,0,1,2,0,1,2] * 10,
                "y_pred2": [0,1,2,0,2,2,0,1,1,0,1,2,0,1,2] * 10,
                "y_pred3": [0,1,2,1,1,2,0,1,2,0,1,2,0,1,2] * 10,
            })
            st.dataframe(example_data.head(10), use_container_width=True)
            csv_example = example_data.to_csv(index=False).encode()
            st.download_button(
                "📥 " + get_text('download_example', lang),
                csv_example,
                "ejemplo_evaluacion_modelos.csv",
                "text/csv"
            )

        origen_csv = st.radio(get_text('select_csv_source', lang), (get_text('upload_from_computer', lang), get_text('select_from_github', lang)))
        csv_file = None

        if origen_csv == get_text('upload_from_computer', lang):
            csv_file = st.file_uploader(
                "📁 " + get_text('load_csv', lang),
                type="csv",
                key="comparison_csv"
            )
        else:
            st.markdown(get_text('searching_csv', lang))
            csvs_disponibles = listar_archivos_github(GITHUB_USER, GITHUB_REPO, GITHUB_IMG_FOLDER, [".csv"])
            if csvs_disponibles:
                csv_elegido = st.selectbox(get_text('choose_csv', lang), csvs_disponibles)
                if st.button(get_text('load_selected_csv', lang)):
                    url_csv = get_github_image_url(GITHUB_USER, GITHUB_REPO, GITHUB_IMG_FOLDER, csv_elegido)
                    response = requests.get(url_csv)
                    if response.status_code == 200:
                        csv_file = io.BytesIO(response.content)
                        st.success(f"{get_text('file_loaded', lang)} '{csv_elegido}'.")
                    else:
                        st.error(get_text('error_loading_csv', lang))
            else:
                st.warning(get_text('no_csv_found', lang))

        if csv_file is not None:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip().str.lower()
            required_cols = ['y_true', 'y_pred1', 'y_pred2', 'y_pred3']
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ {get_text('error_loading_csv', lang)}: {required_cols}")
                st.write(f"Columnas encontradas: {list(df.columns)}")
                st.stop()
            y_true = df['y_true']
            y_pred1 = df['y_pred1']
            y_pred2 = df['y_pred2']
            y_pred3 = df['y_pred3']
            predictions = [y_pred1, y_pred2, y_pred3]
            st.session_state.eval_predictions = predictions
            st.session_state.eval_y_true = y_true
            st.success(f"✅ {get_text('file_loaded', lang)}. {len(df)} {get_text('samples_found', lang)}.")

            metrics = []
            confusion_figs = []
            for i, (y_pred, name) in enumerate(zip(predictions, MODEL_NAMES)):
                advanced_metrics = calculate_advanced_metrics(y_true, y_pred)
                metrics.append(advanced_metrics)
                fig = plot_confusion_matrix_mpl(y_true, y_pred, CLASSES, f"{get_text('model', lang)} - {name}")
                img_base64 = fig_to_base64(fig)
                confusion_figs.append(img_base64)
            metric_df = pd.DataFrame({
                get_text('model', lang): MODEL_NAMES,
                get_text('precision', lang): [m['accuracy'] for m in metrics],
                get_text('recall', lang): [m['report']['macro avg']['recall'] for m in metrics],
                get_text('f1_score', lang): [m['report']['macro avg']['f1-score'] for m in metrics],
                get_text('mcc', lang): [m['mcc'] for m in metrics],
                get_text('avg_specificity', lang): [np.mean(m['specificity']) for m in metrics]
            })
            styled_df = metric_df.style.highlight_max(axis=0)
            st.dataframe(styled_df, use_container_width=True)
            with st.expander("🛈 ¿Qué significa cada métrica?"):
                for k, v in METRIC_DESCRIPTIONS.items():
                    st.markdown(f"- **{k}:** {v}")
            best_idx = int(np.argmax(metric_df[get_text('mcc', lang)]))
            st.success(f"⭐ {get_text('best_model', lang)}: **{MODEL_NAMES[best_idx]}** ({get_text('mcc', lang)}: {metric_df[get_text('mcc', lang)][best_idx]:.4f})")
            st.markdown("#### 📊 " + get_text('statistical_analysis', lang))
            for i in range(len(MODEL_NAMES)):
                for j in range(i+1, len(MODEL_NAMES)):
                    stat, pval = mcnemar_test(y_true, predictions[i], predictions[j])
                    signo = "✅ " + get_text('significant', lang) if pval < 0.05 else "❌ " + get_text('not_significant', lang)
                    st.write(f"{MODEL_NAMES[i]} vs {MODEL_NAMES[j]}: Estadístico = {stat:.3f}, p-valor = {pval:.4f} {signo}")

            st.session_state.eval_metrics = metrics
            st.session_state.eval_confusion_figs = confusion_figs
            st.session_state.eval_model_names = MODEL_NAMES

    with tab2:
        st.markdown("### 📄 " + get_text('generate_comparative_report', lang))
        if ('eval_metrics' not in st.session_state or
            'eval_predictions' not in st.session_state or
            'eval_y_true' not in st.session_state):
            st.warning("⚠️ " + get_text('analysis_first', lang))
        else:
            st.info(get_text('report_includes', lang))
            if st.button("🔄 " + get_text('generate_comparative_report', lang), type="primary"):
                with st.spinner(get_text('generating_report', lang)):
                    try:
                        metrics = st.session_state.eval_metrics
                        confusion_figs = st.session_state.eval_confusion_figs
                        html_report = generate_comparative_report(
                            metrics,
                            confusion_figs,
                            MODEL_NAMES,
                            st.session_state.eval_y_true,
                            st.session_state.eval_predictions,
                            lang
                        )
                        if PDFKIT_AVAILABLE:
                            pdf_bytes = pdfkit.from_string(html_report, False)
                            st.download_button(
                                label=f"📄 {get_text('download_pdf', lang)}",
                                data=pdf_bytes,
                                file_name=f"reporte_comparativo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                            st.success("✅ " + get_text('report_generated', lang))
                        else:
                            generate_html_download(html_report, filename_prefix="reporte_comparativo", lang=lang)
                            st.info(get_text('pdfkit_not_available', lang))
                    except Exception as e:
                        st.error(f"Error generando reporte: {e}")

```
▶️ **Ejecutar**: Crea el archivo app.py con la interfaz web de Streamlit

---

### 🎯 Ejecución de la Aplicación

Una vez completadas todas las celdas:

1. 🌐 Ejecuta la aplicación:
```python
!streamlit run app.py &
```

2. 🔗 Utiliza ngrok para crear un túnel público en caso no aparece el link:
```python
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(port='8501')
print(f"Aplicación disponible en: {public_url}")
```

## 📝 Estructura del Proyecto

```
cacao-disease-detection/
├── 📓 CacaoDeep.ipynb          # Notebook principal
├── 📄 README.md                # Este archivo
├── 🐍 app.py                   # Aplicación Streamlit
├── 🤖 cnn_simple.h5            # Modelo CNN simple
├── 🤖 cnn_deep.h5              # Modelo CNN profundo
├── 🤖 mobilenet.h5             # Modelo MobileNetV2
├── 🤖 resnet50.h5              # Modelo ResNet50
├── 📊 requirements.txt         # Dependencias
└── 📁 cacao_data/              # Dataset descargado
```

## 🎯 Uso de la Aplicación

1. 📤 **Subir imagen**: Arrastra o selecciona una imagen de mazorca de cacao
2. 🔍 **Diagnóstico**: El sistema analiza la imagen con los 4 modelos
3. 📊 **Resultados**: Muestra predicciones, confianza y métricas
4. 🔥 **Mapa de calor**: Visualiza las áreas de atención del modelo
5. 📋 **Reporte**: Genera un reporte PDF con todos los resultados

## 📊 Métricas de Evaluación

- **MCC (Matthews Correlation Coefficient)**: Métrica robusta para datasets desbalanceados
- **Especificidad**: Capacidad de identificar correctamente casos negativos
- **F1-Score**: Balance entre precisión y recall
- **Prueba de McNemar**: Validación estadística de diferencias entre modelos

## 🔬 Metodología

1. **Preprocesamiento**: Normalización, redimensionamiento y aumento de datos
2. **Entrenamiento**: Validación cruzada estratificada con 5 pliegues
3. **Evaluación**: Métricas robustas y validación estadística
4. **Interpretabilidad**: Mapas Grad-CAM para explicar decisiones

## 🚀 Resultados Destacados

- 🏆 **MobileNetV2** logró el mejor rendimiento con MCC = 0.910
- 📈 **Especificidad promedio** de 0.969 en MobileNetV2
- ✅ **Diferencias estadísticamente significativas** validadas con McNemar
- 🎯 **Focalización correcta** en zonas necróticas y bordes miceliales

## 👨‍💻 Autores

- **Hilda Ayde Galdos Jara** - *Universidad Nacional de Trujillo* - hgaldos@unitru.edu.pe
- **Dante Joel Vasquez Rodriguez** - *Universidad Nacional de Trujillo* - dvasquezrod@unitru.edu.pe

## 🙏 Agradecimientos

- 🎓 Ing. Juan Pedro Santos Fernandez
- 📊 Kaggle por el Cocoa Diseases Dataset
- 🤖 Google Colab por el entorno de entrenamiento
- 🌐 Comunidad de TensorFlow y Keras

## 📞 Soporte

Si tienes preguntas o problemas:

1. 🐛 Reporta bugs en [Issues](https://github.com/tu-usuario/cacao-disease-detection/issues)
2. 📧 Contacta a los autores por email
3. 📖 Consulta la documentación del proyecto

---

## 🔗 Enlaces Útiles

- 🎯 [Dataset Original](https://www.kaggle.com/datasets/zaldyjr/cacao-diseases)
- 📚 [Documentación TensorFlow](https://www.tensorflow.org/)
- 🌐 [Documentación Streamlit](https://docs.streamlit.io/)
- 🤖 [Google Colab](https://colab.research.google.com/)

---

⭐ **¡Si este proyecto te fue útil, no olvides darle una estrella!** ⭐
