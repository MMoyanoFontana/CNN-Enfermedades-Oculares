# CNN para Detección de Enfermedades Oculares 👁️

Un proyecto de aprendizaje automático que combina Redes Neuronales Convolucionales (CNN) y algoritmos H2O GBM para la clasificación automática de enfermedades oculares a partir de imágenes.

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de enfermedades oculares utilizando dos enfoques complementarios:

1. **CNN (ConvNeXt)**: Red neuronal convolucional preentrenada para extracción de características visuales
2. **H2O GBM**: Algoritmo de Gradient Boosting para clasificación basada en características extraídas

El sistema es capaz de clasificar las 5 principales enfermedades oculares presentes en el dataset, utilizando técnicas avanzadas de balanceo de clases y validación cruzada.

## 🔧 Características Principales

- **Arquitectura Híbrida**: Combina CNN para extracción de características y GBM para clasificación final
- **Modelo Pre-entrenado**: Utiliza ConvNeXt-Base con fine-tuning especializado
- **Balanceo de Clases**: Manejo automático de datasets desbalanceados
- **Augmentación de Datos**: Transformaciones avanzadas para mejorar la generalización
- **Validación Robusta**: Evaluación completa con matrices de confusión y curvas ROC
- **Reproducibilidad**: Seeds fijas para resultados consistentes

## 📊 Métricas y Visualizaciones

El proyecto genera automáticamente:

- **Matrices de Confusión**: Para CNN y H2O GBM
- **Curvas ROC**: Análisis multi-clase del rendimiento
- **Gráficos de Entrenamiento**: Evolución de pérdida y precisión por época

## 🛠️ Tecnologías Utilizadas

- **PyTorch**: Framework principal para deep learning
- **Torchvision**: Transformaciones y modelos pre-entrenados
- **TIMM**: Biblioteca de modelos de visión por computadora
- **H2O**: Plataforma de machine learning para GBM
- **Scikit-learn**: Métricas y división de datos
- **Matplotlib/Seaborn**: Visualización de resultados
- **Pandas/NumPy**: Manipulación de datos

## 📁 Estructura del Proyecto

```
CNN-Enfermedades-Oculares/
├── h2o_ojos_v3.py                # Script principal
├── CurvaROC_CNN.png              # Curva ROC del modelo CNN
├── CurvaROC_CNN_Validacion.png   # Curva ROC de validación CNN
├── CurvaROC_H2O.png              # Curva ROC del modelo H2O
├── MatrizConfusionFinal_CNN.png  # Matriz de confusión CNN
├── MatrizConfusionFinal_H2O.png  # Matriz de confusión H2O
└── Plot_CNN_Entrenamiento.png    # Gráficos de entrenamiento
```

## 🚀 Instalación y Uso


### Ejecución

1. **Clona el repositorio:**
```bash
git clone https://github.com/MMoyanoFontana/CNN-Enfermedades-Oculares.git
pip install -r requirements.txt
cd CNN-Enfermedades-Oculares
```

2. **Ejecuta el script principal:**
```bash
python h2o_ojos_v3.py
```

El script automáticamente:
- Descarga el dataset si no está presente
- Entrena el modelo CNN (si no existe)
- Extrae características usando la CNN
- Entrena el modelo H2O GBM
- Genera todas las visualizaciones

## 📈 Dataset

El proyecto utiliza el **Eye Disease Image Dataset** de Kaggle, que contiene imágenes de alta calidad de diferentes condiciones oculares. El sistema selecciona automáticamente las 5 clases principales basándose en la cantidad de muestras disponibles.

### Características del Dataset:
- **Fuente**: Kaggle - Eye Disease Image Dataset
- **Formato**: Imágenes RGB de alta resolución
- **Clases**: Múltiples enfermedades oculares
- **Tamaño**: Variable según disponibilidad

## ⚙️ Configuración

Las principales configuraciones se pueden modificar en el archivo `h2o_ojos_v3.py`:

```python
CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR = 5  # Número de clases a clasificar
SEMILLA_ALEATORIA = 42                          # Seed para reproducibilidad
```

### Parámetros del Modelo CNN:
- **Arquitectura**: ConvNeXt-Base
- **Optimizador**: AdamW con learning rate 1e-4
- **Scheduler**: ReduceLROnPlateau
- **Épocas**: 20 (con early stopping)
- **Batch Size**: 16

### Parámetros del Modelo H2O GBM:
- **Árboles**: 500
- **Learning Rate**: 0.05
- **Profundidad Máxima**: 10
- **Validación Cruzada**: 5-fold

## 📊 Resultados

El sistema proporciona evaluaciones detalladas incluyendo:

- **Precisión del modelo CNN** en todas las imágenes de validación
- **Precisión del modelo H2O GBM** comparativa
- **Métricas por clase** individuales
- **Visualizaciones interactivas** de rendimiento

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---