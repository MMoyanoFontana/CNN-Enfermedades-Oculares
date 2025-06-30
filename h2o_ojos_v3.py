import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, \
    Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from collections import defaultdict, Counter
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# --- H2O Imports ---
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import pandas as pd  # For easier conversion to H2OFrame

# --- Configuración Global ---
CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR = 5
SEMILLA_ALEATORIA = 42
random.seed(SEMILLA_ALEATORIA)
np.random.seed(SEMILLA_ALEATORIA)
torch.manual_seed(SEMILLA_ALEATORIA)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA_ALEATORIA)

# --- Ruta al Dataset ---
directorio_base_dataset = "./ojos/Augmented Dataset/Augmented Dataset"
if not os.path.isdir(directorio_base_dataset):
    print(f"Directorio del dataset no encontrado: {directorio_base_dataset}")
    try:
        import kagglehub

        print("Intentando descargar/verificar el dataset con kagglehub...")
        ruta_raiz_descarga = kagglehub.dataset_download("ruhulaminsharif/eye-disease-image-dataset")
        directorio_base_dataset = os.path.join(ruta_raiz_descarga, 'Augmented Dataset', 'Augmented Dataset')
        print(f"Dataset descargado/verificado en: {directorio_base_dataset}")
        if not os.path.isdir(directorio_base_dataset):
            print(f"Error: La estructura esperada del dataset no se encontró en {directorio_base_dataset}")
            exit()
    except Exception as e:
        print(f"Error al descargar o configurar el dataset con kagglehub: {e}")
        print("Por favor, descargá manualmente y configurá 'directorio_base_dataset'.")
        exit()
ruta_dataset = directorio_base_dataset
# --- Fin Ruta al Dataset ---

ruta_modelo_cnn = f"ojos_convnext_base_cnn_top{CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR}_balanceado_mejor_modelo.pth"
ruta_modelo_h2o = f"ojos_h2o_gbm_top{CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR}_mejor_modelo.zip"


def obtener_dispositivo():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")
    return dispositivo


def obtener_transformaciones():
    return {
        'entrenamiento': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'validacion': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }


class DatasetGenericoDesdeMuestras(Dataset):
    def __init__(self, lista_info_muestras, transformacion, cargador_imagenes_fn):
        self.lista_info_muestras = lista_info_muestras
        self.transformacion = transformacion
        self.cargador_imagenes = cargador_imagenes_fn
        self.etiquetas = [info[1] for info in self.lista_info_muestras]

    def __len__(self):
        return len(self.lista_info_muestras)

    def __getitem__(self, idx):
        ruta_imagen, etiqueta = self.lista_info_muestras[idx]
        imagen = self.cargador_imagenes(ruta_imagen)
        if self.transformacion:
            imagen = self.transformacion(imagen)
        return imagen, etiqueta


def obtener_info_muestras_filtradas_SIN_balanceo(directorio_raiz, cantidad_clases_principales,
                                                 semilla_aleatoria_actual):
    if semilla_aleatoria_actual is not None:
        random.seed(semilla_aleatoria_actual)

    dataset_temporal_completo = datasets.ImageFolder(directorio_raiz)
    nombres_clases_originales = dataset_temporal_completo.classes

    conteos_clases = Counter(dataset_temporal_completo.targets)
    clases_ordenadas_por_conteo = sorted(conteos_clases.items(), key=lambda item: (item[1], item[0]), reverse=True)

    if len(clases_ordenadas_por_conteo) < cantidad_clases_principales:
        print(
            f"Advertencia: El dataset tiene {len(clases_ordenadas_por_conteo)} clases, menos que las {cantidad_clases_principales} solicitadas. Se usarán todas las clases disponibles.")
        cantidad_clases_principales = len(clases_ordenadas_por_conteo)
        if cantidad_clases_principales == 0:
            raise ValueError("No se encontraron clases en el dataset.")

    indices_originales_top_n = [idx_original for idx_original, conteo in
                                clases_ordenadas_por_conteo[:cantidad_clases_principales]]

    print(f"Clases originales totales: {len(nombres_clases_originales)}")
    print(f"Top {cantidad_clases_principales} clases seleccionadas (índices originales): {indices_originales_top_n}")
    print("NO se realizará balanceo de clases; se usarán todas las imágenes disponibles de las clases seleccionadas.")

    nombres_clases_seleccionadas = [nombres_clases_originales[i] for i in indices_originales_top_n]
    mapa_idxoriginal_a_idxnuevo = {idx_orig: idx_nuevo for idx_nuevo, idx_orig in enumerate(indices_originales_top_n)}

    lista_muestras_final = []
    muestras_por_clase_original = defaultdict(list)
    for ruta, etiqueta_original in dataset_temporal_completo.samples:
        muestras_por_clase_original[etiqueta_original].append(ruta)

    for idx_clase_original in indices_originales_top_n:
        nueva_etiqueta = mapa_idxoriginal_a_idxnuevo[idx_clase_original]
        rutas_imagenes_clase = muestras_por_clase_original[idx_clase_original]
        nombre_clase_actual = nombres_clases_originales[idx_clase_original]
        cantidad_disponible = len(rutas_imagenes_clase)

        if cantidad_disponible == 0:
            print(
                f"Advertencia: La clase original '{nombre_clase_actual}' (idx {idx_clase_original}) no tiene imágenes. Se omitirá.")
            continue

        # Sin balanceo, se toman todas las imágenes de la clase
        rutas_seleccionadas = rutas_imagenes_clase

        # print(f"  Clase '{nombre_clase_actual}': usando {len(rutas_seleccionadas)} imágenes.") # Silenciar este print
        for ruta_img in rutas_seleccionadas:
            lista_muestras_final.append((ruta_img, nueva_etiqueta))

    if not lista_muestras_final:
        raise ValueError(
            "No se seleccionaron muestras. Verificá la lógica de 'obtener_info_muestras_filtradas_SIN_balanceo'.")

    if semilla_aleatoria_actual is not None:
        random.seed(semilla_aleatoria_actual)
    random.shuffle(lista_muestras_final)

    return lista_muestras_final, nombres_clases_seleccionadas


def cargar_conjuntos_de_datos(directorio_datos, dict_transformaciones, cantidad_clases_principales,
                              tamaño_split_test=0.2, semilla_aleatoria_split=42,
                              return_all_samples_for_evaluation=False):
    """
    Carga los conjuntos de datos de entrenamiento y validación.
    Si return_all_samples_for_evaluation es True, devuelve un solo dataset con todas las muestras sin split ni balanceo.
    """
    if return_all_samples_for_evaluation:
        lista_muestras_base, nombres_clases = obtener_info_muestras_filtradas_SIN_balanceo(
            directorio_datos, cantidad_clases_principales, semilla_aleatoria_split
        )
        if not lista_muestras_base:
            print("Error: No se cargaron muestras base para evaluación completa.")
            return None, None, [], []
        dataset_completo = DatasetGenericoDesdeMuestras(lista_muestras_base,
                                                        dict_transformaciones['validacion'],
                                                        # Usar transformaciones de validación
                                                        datasets.folder.default_loader)
        return dataset_completo, None, nombres_clases, dataset_completo.etiquetas  # Retorna solo el dataset completo y las etiquetas

    # Lógica original para entrenamiento/validación (con balanceo)
    lista_muestras_base, nombres_clases = obtener_info_muestras_filtradas_balanceadas(  # Usa la función con balanceo
        directorio_datos, cantidad_clases_principales, semilla_aleatoria_split
    )

    if not lista_muestras_base:
        print("Error: No se cargaron muestras base.")
        return None, None, [], []

    etiquetas_para_split = [muestra[1] for muestra in lista_muestras_base]

    if not etiquetas_para_split:
        print("Error: No hay etiquetas para realizar el split.")
        return None, None, [], []

    if len(set(etiquetas_para_split)) < 2 and tamaño_split_test > 0 and len(etiquetas_para_split) > 0:
        print(
            f"Advertencia: Solo hay {len(set(etiquetas_para_split))} clase(s) después del filtrado. No se puede hacer split estratificado para validación.")
        if tamaño_split_test > 0:
            print("Usando todo como entrenamiento, no habrá conjunto de validación.")
        indices_entrenamiento_locales = list(range(len(etiquetas_para_split)))
        indices_validacion_locales = np.array([])  # Ensure it's an array for .size
    elif tamaño_split_test == 0 or len(etiquetas_para_split) == 0:
        indices_entrenamiento_locales = list(range(len(etiquetas_para_split)))
        indices_validacion_locales = np.array([])
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=tamaño_split_test, random_state=semilla_aleatoria_split)
        indices_entrenamiento_locales, indices_validacion_locales = next(
            sss.split(np.arange(len(etiquetas_para_split)), etiquetas_para_split))

    muestras_entrenamiento_info = [lista_muestras_base[i] for i in indices_entrenamiento_locales]
    dataset_entrenamiento = DatasetGenericoDesdeMuestras(muestras_entrenamiento_info,
                                                         dict_transformaciones['entrenamiento'],
                                                         datasets.folder.default_loader)
    dataset_validacion = None
    if indices_validacion_locales.size > 0:
        muestras_validacion_info = [lista_muestras_base[i] for i in indices_validacion_locales]
        dataset_validacion = DatasetGenericoDesdeMuestras(muestras_validacion_info,
                                                          dict_transformaciones['validacion'],
                                                          datasets.folder.default_loader)
    etiquetas_entrenamiento = dataset_entrenamiento.etiquetas
    return dataset_entrenamiento, dataset_validacion, nombres_clases, etiquetas_entrenamiento


def obtener_info_muestras_filtradas_balanceadas(directorio_raiz, cantidad_clases_principales, semilla_aleatoria_actual):
    """
    Obtiene la información de las muestras filtradas por las N clases principales y balanceadas.
    Esta es la función usada para el entrenamiento de la CNN.
    """
    if semilla_aleatoria_actual is not None:
        random.seed(semilla_aleatoria_actual)

    dataset_temporal_completo = datasets.ImageFolder(directorio_raiz)
    nombres_clases_originales = dataset_temporal_completo.classes

    conteos_clases = Counter(dataset_temporal_completo.targets)
    clases_ordenadas_por_conteo = sorted(conteos_clases.items(), key=lambda item: (item[1], item[0]), reverse=True)

    if len(clases_ordenadas_por_conteo) < cantidad_clases_principales:
        print(
            f"Advertencia: El dataset tiene {len(clases_ordenadas_por_conteo)} clases, menos que las {cantidad_clases_principales} solicitadas. Se usarán todas las clases disponibles.")
        cantidad_clases_principales = len(clases_ordenadas_por_conteo)
        if cantidad_clases_principales == 0:
            raise ValueError("No se encontraron clases en el dataset.")

    indices_originales_top_n = [idx_original for idx_original, conteo in
                                clases_ordenadas_por_conteo[:cantidad_clases_principales]]
    cantidad_balanceo = clases_ordenadas_por_conteo[cantidad_clases_principales - 1][1]

    print(f"Clases originales totales: {len(nombres_clases_originales)}")
    print(f"Top {cantidad_clases_principales} clases seleccionadas (índices originales): {indices_originales_top_n}")
    print(f"Balanceando con {cantidad_balanceo} imágenes por clase seleccionada.")

    nombres_clases_seleccionadas = [nombres_clases_originales[i] for i in indices_originales_top_n]
    mapa_idxoriginal_a_idxnuevo = {idx_orig: idx_nuevo for idx_nuevo, idx_orig in enumerate(indices_originales_top_n)}

    lista_muestras_final = []
    muestras_por_clase_original = defaultdict(list)
    for ruta, etiqueta_original in dataset_temporal_completo.samples:
        muestras_por_clase_original[etiqueta_original].append(ruta)

    for idx_clase_original in indices_originales_top_n:
        nueva_etiqueta = mapa_idxoriginal_a_idxnuevo[idx_clase_original]
        rutas_imagenes_clase = muestras_por_clase_original[idx_clase_original]
        nombre_clase_actual = nombres_clases_originales[idx_clase_original]
        cantidad_disponible = len(rutas_imagenes_clase)

        if cantidad_disponible == 0:
            print(
                f"Advertencia: La clase original '{nombre_clase_actual}' (idx {idx_clase_original}) no tiene imágenes. Se omitirá.")
            continue

        if cantidad_disponible < cantidad_balanceo:
            print(
                f"Advertencia: Clase '{nombre_clase_actual}' (idx {idx_clase_original}) tiene {cantidad_disponible} imágenes, menos que el objetivo de balanceo {cantidad_balanceo}. Se usarán todas las {cantidad_disponible} imágenes.")
            rutas_seleccionadas = rutas_imagenes_clase
        else:
            rutas_seleccionadas = random.sample(rutas_imagenes_clase, cantidad_balanceo)

        for ruta_img in rutas_seleccionadas:
            lista_muestras_final.append((ruta_img, nueva_etiqueta))

    if not lista_muestras_final:
        raise ValueError(
            "No se seleccionaron muestras. Verificá la lógica de 'obtener_info_muestras_filtradas_balanceadas'.")

    if semilla_aleatoria_actual is not None:
        random.seed(semilla_aleatoria_actual)
    random.shuffle(lista_muestras_final)

    return lista_muestras_final, nombres_clases_seleccionadas


def construir_modelo_cnn(numero_clases, dispositivo, congelar_capas=True):
    modelo = timm.create_model('convnext_base', pretrained=True, num_classes=numero_clases)
    if congelar_capas:
        for nombre_param, parametro in modelo.named_parameters():
            if nombre_param.startswith('head.'):
                parametro.requires_grad = True
            elif "stages.2" in nombre_param or "stages.3" in nombre_param:
                parametro.requires_grad = True
            else:
                parametro.requires_grad = False
    else:
        for parametro in modelo.parameters():
            parametro.requires_grad = True
    return modelo.to(dispositivo)


def entrenar_una_epoca(modelo, cargador_datos, criterio, optimizador, dispositivo, escalador_gradiente):
    modelo.train()
    perdida_acumulada, correctas, total_muestras = 0.0, 0, 0
    for imagenes, etiquetas in tqdm(cargador_datos, desc="Entrenando CNN", leave=False):
        imagenes, etiquetas = imagenes.to(dispositivo, non_blocking=True), etiquetas.to(dispositivo, non_blocking=True)
        optimizador.zero_grad(set_to_none=True)
        if dispositivo.type == 'cuda' and escalador_gradiente:
            with autocast(device_type='cuda'):
                salidas = modelo(imagenes)
                perdida = criterio(salidas, etiquetas)
            escalador_gradiente.scale(perdida).backward()
            escalador_gradiente.step(optimizador)
            escalador_gradiente.update()
        else:
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            perdida.backward()
            optimizador.step()
        perdida_acumulada += perdida.item() * imagenes.size(0)
        _, predicciones = torch.max(salidas, 1)
        correctas += (predicciones == etiquetas).sum().item()
        total_muestras += etiquetas.size(0)
    perdida_epoca = perdida_acumulada / total_muestras
    precision_epoca = correctas / total_muestras
    return perdida_epoca, precision_epoca


def validar_cnn(modelo, cargador_datos, criterio, dispositivo):
    modelo.eval()
    perdida_acumulada, correctas, total_muestras = 0.0, 0, 0
    if cargador_datos is None:
        return 0.0, 0.0, [], []
    all_labels = []
    all_preds_proba = []
    with torch.no_grad():
        for imagenes, etiquetas in tqdm(cargador_datos, desc="Validando CNN", leave=False):
            imagenes, etiquetas = imagenes.to(dispositivo, non_blocking=True), etiquetas.to(dispositivo,
                                                                                            non_blocking=True)
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            perdida_acumulada += perdida.item() * imagenes.size(0)
            _, predicciones = torch.max(salidas, 1)
            correctas += (predicciones == etiquetas).sum().item()
            total_muestras += etiquetas.size(0)

            # Para la curva ROC
            all_labels.extend(etiquetas.cpu().numpy())
            # Aplicar softmax para obtener probabilidades si el criterio no lo hace ya
            probs = torch.nn.functional.softmax(salidas, dim=1)
            all_preds_proba.extend(probs.cpu().numpy())

    if total_muestras == 0: return 0.0, 0.0, [], []
    perdida_epoca = perdida_acumulada / total_muestras
    precision_epoca = correctas / total_muestras
    return perdida_epoca, precision_epoca, all_labels, all_preds_proba


def graficar_metricas(precision_entrenamiento, precision_validacion, perdida_entrenamiento, perdida_validacion, epocas,
                      prefijo_titulo="CNN", save_filename=""):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(epocas), precision_entrenamiento, label='Entrenamiento')
    if precision_validacion and any(
            p is not None and p != 0.0 for p in precision_validacion):  # Check for actual validation data
        plt.plot(range(epocas), precision_validacion, label='Validación')
    plt.title(f'{prefijo_titulo} - Precisión por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(epocas), perdida_entrenamiento, label='Entrenamiento')
    if perdida_validacion and any(
            p is not None and p != 0.0 for p in perdida_validacion):  # Check for actual validation data
        plt.plot(range(epocas), perdida_validacion, label='Validación')
    plt.title(f'{prefijo_titulo} - Pérdida por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename)
    plt.show()


def graficar_roc(y_true, y_score, n_classes, class_names, save_filename=""):
    plt.figure(figsize=(10, 8))
    # Calcular la curva ROC y el AUC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Graficar la curva ROC para cada clase
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC de clase {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC multi-clase')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_filename:
        plt.savefig(save_filename)
    plt.show()


def construir_extractor_caracteristicas(ruta_modelo_cnn_entrenado, numero_clases_seleccionadas, dispositivo):
    modelo = timm.create_model('convnext_base', pretrained=False, num_classes=numero_clases_seleccionadas)
    modelo.load_state_dict(torch.load(ruta_modelo_cnn_entrenado, map_location=dispositivo, weights_only=True),
                           strict=True)
    # Reemplazar la capa de clasificación con una capa de pooling global y flatten
    # para extraer características antes de la capa final
    modelo.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return modelo.to(dispositivo).eval()


def extraer_caracteristicas_desde_cargador(modelo_extractor, cargador_datos, dispositivo):
    lista_caracteristicas = []
    lista_etiquetas = []
    modelo_extractor.eval()
    if cargador_datos is None: return np.array([]), np.array([])
    with torch.no_grad():
        for imagenes, etiquetas in tqdm(cargador_datos, desc="Extrayendo Características", leave=False):
            imagenes = imagenes.to(dispositivo, non_blocking=True)
            caracteristicas_actuales = modelo_extractor(imagenes)
            lista_caracteristicas.append(caracteristicas_actuales.cpu().numpy())
            lista_etiquetas.append(etiquetas.cpu().numpy())
    if not lista_caracteristicas: return np.array([]), np.array([])
    return np.concatenate(lista_caracteristicas), np.concatenate(lista_etiquetas)


def evaluar_precision_final_cnn(modelo_cnn, directorio_datos, dict_transformaciones, cantidad_clases, dispositivo,
                                nombres_clases):
    print("\nEvaluando la precisión final del modelo CNN en TODAS las imágenes de las clases válidas (sin balanceo)...")

    # Cargar todas las muestras de las clases seleccionadas sin balanceo
    dataset_completo_cnn, _, nombres_clases_eval, etiquetas_completas_eval = cargar_conjuntos_de_datos(
        directorio_datos,
        dict_transformaciones,  # Usar dict_transformaciones completo, la función interna elige 'validacion'
        cantidad_clases_principales=cantidad_clases,
        return_all_samples_for_evaluation=True,  # Indicador para cargar todas las muestras sin split/balanceo
        semilla_aleatoria_split=SEMILLA_ALEATORIA
    )

    if dataset_completo_cnn is None or len(dataset_completo_cnn) == 0:
        print("No hay datos para evaluar la precisión final de la CNN.")
        return

    batch_size_eval = 32
    num_workers_eval = 2 if os.name == 'nt' else 4
    cargador_evaluacion = DataLoader(dataset_completo_cnn, batch_size=batch_size_eval, shuffle=False,
                                     num_workers=num_workers_eval, pin_memory=True,
                                     persistent_workers=num_workers_eval > 0)

    modelo_cnn.eval()
    all_true_labels = []
    all_predictions = []
    all_probabilities = []  # Para la curva ROC
    correctas = 0
    total_muestras = 0

    with torch.no_grad():
        for imagenes, etiquetas in tqdm(cargador_evaluacion, desc="Evaluando Precisión Final CNN", leave=False):
            imagenes, etiquetas = imagenes.to(dispositivo, non_blocking=True), etiquetas.to(dispositivo,
                                                                                            non_blocking=True)
            salidas = modelo_cnn(imagenes)
            _, predicciones = torch.max(salidas, 1)

            all_true_labels.extend(etiquetas.cpu().numpy())
            all_predictions.extend(predicciones.cpu().numpy())
            all_probabilities.extend(torch.nn.functional.softmax(salidas, dim=1).cpu().numpy())

            correctas += (predicciones == etiquetas).sum().item()
            total_muestras += etiquetas.size(0)

    precision_final = correctas / total_muestras
    print(f"\n--- Resultados de la Evaluación Final de la CNN ---")
    print(f"Total de imágenes evaluadas: {total_muestras}")
    print(f"Imágenes clasificadas correctamente: {correctas}")
    print(f"Precisión FINAL del modelo CNN en todas las imágenes válidas: {precision_final:.4f}")

    # Generar y mostrar la matriz de confusión
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.title(f'Matriz de Confusión CNN (Todas las imágenes de las {len(nombres_clases)} clases principales)')
    plt.savefig("MatrizConfusionFinal_CNN.png")
    plt.show()

    # Generar y mostrar la curva ROC
    graficar_roc(np.array(all_true_labels), np.array(all_probabilities), len(nombres_clases), nombres_clases,
                 save_filename="CurvaROC_CNN.png")


def ejecutar_entrenamiento_cnn_si_necesario(dispositivo, dict_transformaciones, ref_nombres_clases,
                                            ref_etiquetas_entrenamiento_cnn):
    modelo_cnn_cargado = None
    if os.path.exists(ruta_modelo_cnn):
        print(f"Modelo CNN encontrado en {ruta_modelo_cnn}. Salteando entrenamiento de CNN.")
        # Se necesita `nombres_clases` para inicializar el modelo correctamente,
        # incluso si no se entrena ahora. Se obtiene con la función de balanceo
        # ya que es la que define la lista de clases finales para el entrenamiento.
        _, nombres_clases_cargados = obtener_info_muestras_filtradas_balanceadas(
            ruta_dataset, CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR, SEMILLA_ALEATORIA
        )
        ref_nombres_clases[:] = nombres_clases_cargados

        # Cargar el modelo para evaluación posterior
        modelo_cnn_cargado = construir_modelo_cnn(len(ref_nombres_clases), dispositivo, congelar_capas=False)
        modelo_cnn_cargado.load_state_dict(torch.load(ruta_modelo_cnn, map_location=dispositivo), strict=True)
        modelo_cnn_cargado.eval()  # Poner en modo evaluación
        return modelo_cnn_cargado

    print("Modelo CNN no encontrado. Iniciando entrenamiento de CNN.")
    print(f"Cargando datasets para CNN (Top {CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR} clases, balanceado)...")
    dataset_entrenamiento, dataset_validacion, nombres_clases_cargados, etiquetas_entrenamiento_cargadas = cargar_conjuntos_de_datos(
        ruta_dataset, dict_transformaciones,
        cantidad_clases_principales=CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR,
        tamaño_split_test=0.2, semilla_aleatoria_split=SEMILLA_ALEATORIA,
        return_all_samples_for_evaluation=False  # Asegurarse de que no se use el modo de evaluación completa
    )

    if dataset_entrenamiento is None:
        print("Error: No se pudo cargar el dataset de entrenamiento para la CNN.")
        return None

    ref_nombres_clases[:] = nombres_clases_cargados
    ref_etiquetas_entrenamiento_cnn[:] = etiquetas_entrenamiento_cargadas

    print(f"Clases para CNN (Top {CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR}): {ref_nombres_clases}")
    print(f"Número de clases para CNN: {len(ref_nombres_clases)}")

    conteos_clases_entrenamiento = Counter(ref_etiquetas_entrenamiento_cnn)
    print("Distribución de clases en el conjunto de entrenamiento (CNN):")
    for idx_nueva_etiqueta, conteo in sorted(conteos_clases_entrenamiento.items()):
        print(f"  {ref_nombres_clases[idx_nueva_etiqueta]} (Nuevo idx {idx_nueva_etiqueta}): {conteo} imágenes")

    nuevas_etiquetas_unicas = np.unique(ref_etiquetas_entrenamiento_cnn)
    if len(nuevas_etiquetas_unicas) > 1:
        pesos_clases = compute_class_weight('balanced', classes=nuevas_etiquetas_unicas,
                                            y=ref_etiquetas_entrenamiento_cnn)
        tensor_pesos = torch.FloatTensor(pesos_clases).to(dispositivo)
        criterio = nn.CrossEntropyLoss(weight=tensor_pesos)
        print(f"Pesos de clase para CNN (aplicados al criterio): {tensor_pesos.cpu().numpy()}")
    else:
        criterio = nn.CrossEntropyLoss()
        print("Usando CrossEntropyLoss sin pesos de clase (solo 1 clase o error).")

    batch_size = 16
    num_workers = 2 if os.name == 'nt' else 4
    cargador_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    cargador_validacion = None
    if dataset_validacion and len(dataset_validacion) > 0:
        cargador_validacion = DataLoader(dataset_validacion, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    else:
        print("No hay conjunto de validación para la CNN.")

    modelo_cnn = construir_modelo_cnn(len(ref_nombres_clases), dispositivo)
    optimizador = optim.AdamW(filter(lambda p: p.requires_grad, modelo_cnn.parameters()), lr=1e-4, weight_decay=1e-4)
    planificador_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='max', factor=0.5, patience=3,
                                                           verbose=True)
    escalador_gradiente_amp = None
    if dispositivo.type == 'cuda':
        escalador_gradiente_amp = GradScaler(device='cuda')

    cantidad_epocas = 20
    paciencia_early_stopping = 7
    epocas_sin_mejora = 0
    hist_precision_entrenamiento, hist_precision_validacion = [], []
    hist_perdida_entrenamiento, hist_perdida_validacion = [], []
    mejor_precision_validacion = 0.0

    print("Iniciando entrenamiento de la CNN...")
    try:
        for epoca in range(cantidad_epocas):
            print(f"\nÉpoca {epoca + 1}/{cantidad_epocas} (CNN)")
            perdida_ent, precision_ent = entrenar_una_epoca(modelo_cnn, cargador_entrenamiento, criterio, optimizador,
                                                            dispositivo, escalador_gradiente_amp)
            perdida_val_actual, precision_val_actual, _, _ = 0.0, 0.0, [], [] # Reset ROC data for each epoch
            if cargador_validacion:
                perdida_val_actual, precision_val_actual, val_true_labels, val_pred_probabilities = validar_cnn(modelo_cnn, cargador_validacion, criterio,
                                                                       dispositivo)

            hist_perdida_entrenamiento.append(perdida_ent)
            hist_precision_entrenamiento.append(precision_ent)
            hist_perdida_validacion.append(perdida_val_actual if cargador_validacion else None)
            hist_precision_validacion.append(precision_val_actual if cargador_validacion else None)

            print(f"CNN Entrenamiento - Pérdida: {perdida_ent:.4f} | Precisión: {precision_ent:.4f}")
            if cargador_validacion:
                print(f"CNN Validación  - Pérdida: {perdida_val_actual:.4f} | Precisión: {precision_val_actual:.4f}")
                planificador_lr.step(precision_val_actual)
                if precision_val_actual > mejor_precision_validacion:
                    mejor_precision_validacion = precision_val_actual
                    torch.save(modelo_cnn.state_dict(), ruta_modelo_cnn)
                    print(
                        f">> Nuevo mejor modelo CNN guardado con precisión {mejor_precision_validacion:.4f} en {ruta_modelo_cnn}")
                    epocas_sin_mejora = 0
                else:
                    epocas_sin_mejora += 1
                    print(f"No hubo mejora en validación CNN por {epocas_sin_mejora} épocas.")
            else:
                torch.save(modelo_cnn.state_dict(), ruta_modelo_cnn)
                print(f">> Modelo CNN de la época {epoca + 1} guardado (sin set de validación).")

            if cargador_validacion and epocas_sin_mejora >= paciencia_early_stopping:
                print(
                    f">> Deteniendo entrenamiento CNN por early stopping (paciencia: {paciencia_early_stopping} épocas).")
                break
    except KeyboardInterrupt:
        print("\nEntrenamiento CNN interrumpido por el usuario. ¡Qué cagada!")

    if hist_precision_entrenamiento:
        graficar_metricas(hist_precision_entrenamiento, hist_precision_validacion,
                          hist_perdida_entrenamiento, hist_perdida_validacion,
                          len(hist_precision_entrenamiento), prefijo_titulo="CNN",
                          save_filename="Plot_CNN_Entrenamiento.png")

    if cargador_validacion and 'val_true_labels' in locals() and 'val_pred_probabilities' in locals():
        # Graficar ROC para la última época de validación o la mejor época
        print("\nGenerando curva ROC para el conjunto de validación de la CNN...")
        graficar_roc(np.array(val_true_labels), np.array(val_pred_probabilities),
                     len(ref_nombres_clases), ref_nombres_clases,
                     save_filename="CurvaROC_CNN_Validacion.png")

    print(f"Modelo CNN final (o el mejor guardado) está en {ruta_modelo_cnn}")

    # Cargar el mejor modelo guardado para retornarlo
    modelo_cnn_cargado_final = construir_modelo_cnn(len(ref_nombres_clases), dispositivo, congelar_capas=False)
    modelo_cnn_cargado_final.load_state_dict(torch.load(ruta_modelo_cnn, map_location=dispositivo), strict=True)
    modelo_cnn_cargado_final.eval()
    return modelo_cnn_cargado_final


def entrenar_h2o_gbm(X_train, y_train, X_test, y_test, nombres_clases, ruta_guardado_modelo):
    print("\nIniciando entrenamiento del modelo H2O GBM...")
    # Convertir a H2OFrame
    train_df = pd.DataFrame(X_train)
    train_df['target'] = [nombres_clases[i] for i in y_train]
    train_h2o = h2o.H2OFrame(train_df)

    test_df = pd.DataFrame(X_test)
    test_df['target'] = [nombres_clases[i] for i in y_test]
    test_h2o = h2o.H2OFrame(test_df)

    x = train_h2o.col_names[:-1]
    y = 'target'

    # Asegurarse de que la columna 'target' sea de tipo categórico
    train_h2o[y] = train_h2o[y].asfactor()
    test_h2o[y] = test_h2o[y].asfactor()

    modelo_gbm = H2OGradientBoostingEstimator(
        seed=SEMILLA_ALEATORIA,
        nfolds=5,
        ntrees=500,
        learn_rate=0.05,
        max_depth=10,
        stopping_rounds=5,
        stopping_metric="mean_per_class_error", # Usar la métrica correcta para precisión multi-clase
        stopping_tolerance=0.001,
        score_tree_interval=10
    )

    modelo_gbm.train(x=x, y=y, training_frame=train_h2o, validation_frame=test_h2o)

    print("\nEntrenamiento H2O GBM finalizado.")

    modelo_path = h2o.save_model(model=modelo_gbm, path="./", force=True)
    print(f"Modelo H2O GBM guardado en: {modelo_path}")

    if os.path.exists(modelo_path):
        os.rename(modelo_path, ruta_guardado_modelo)
        print(f"Modelo H2O GBM renombrado a: {ruta_guardado_modelo}")

    return modelo_gbm


def evaluar_h2o_gbm(modelo_gbm, X_eval, y_eval, nombres_clases):
    print("\nEvaluando el modelo H2O GBM...")
    eval_df = pd.DataFrame(X_eval)
    eval_df['target'] = [nombres_clases[i] for i in y_eval]
    eval_h2o = h2o.H2OFrame(eval_df)
    eval_h2o['target'] = eval_h2o['target'].asfactor()

    predicciones_h2o = modelo_gbm.predict(eval_h2o)

    # Obtener las etiquetas verdaderas y las predichas
    y_true_h2o = eval_h2o['target'].asnumeric().as_data_frame().values.flatten().astype(int)
    y_pred_h2o = predicciones_h2o['predict'].asnumeric().as_data_frame().values.flatten().astype(int)

    # Obtener las probabilidades predichas para la curva ROC
    prob_cols = nombres_clases # nombres_clases_globales should be passed here, which contains actual class names
    y_prob_h2o = predicciones_h2o[prob_cols].as_data_frame().values

    # Calcular y mostrar la precisión
    precision = (y_true_h2o == y_pred_h2o).mean()
    print(f"Precisión del modelo H2O GBM: {precision:.4f}")

    # Generar y mostrar la matriz de confusión
    cm = confusion_matrix(y_true_h2o, y_pred_h2o)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.title(f'Matriz de Confusión H2O GBM (Todas las imágenes de las {len(nombres_clases)} clases principales)')
    plt.savefig("MatrizConfusionFinal_H2O.png")
    plt.show()

    # Generar y mostrar la curva ROC
    graficar_roc(y_true_h2o, y_prob_h2o, len(nombres_clases), nombres_clases,
                 save_filename="CurvaROC_H2O.png")


# --- Script Principal ---
if __name__ == '__main__':
    # --- H2O Initialization ---
    h2o.init(nthreads=-1, max_mem_size="8G")  # Usar todos los cores, limitar memoria si es necesario
    # Comentamos la siguiente línea para permitir que H2O muestre sus barras de progreso
    # h2o.no_progress()

    dispositivo_torch = obtener_dispositivo()
    transformaciones = obtener_transformaciones()

    nombres_clases_globales = []
    etiquetas_entrenamiento_cnn_globales = []

    # Se retorna el modelo CNN entrenado o cargado
    modelo_cnn_final = ejecutar_entrenamiento_cnn_si_necesario(dispositivo_torch, transformaciones,
                                                                nombres_clases_globales,
                                                                etiquetas_entrenamiento_cnn_globales)

    if modelo_cnn_final is None:
        print("No se pudo obtener el modelo CNN. Saliendo.")
        h2o.cluster().shutdown()
        exit()

    # Evaluación final de la CNN en todas las imágenes válidas
    evaluar_precision_final_cnn(modelo_cnn_final, ruta_dataset, transformaciones,
                                CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR,
                                dispositivo_torch, nombres_clases_globales)

    # --- Preparación para H2O ---
    print("\nPreparando datos para H2O...")

    # Cargar los datos para extracción de características, utilizando las mismas clases seleccionadas
    # Usaremos el dataset completo (sin balanceo) para tener un conjunto de datos representativo
    # para entrenar y evaluar el modelo H2O.
    dataset_completo_para_features, _, _, _ = cargar_conjuntos_de_datos(
        ruta_dataset,
        transformaciones,
        cantidad_clases_principales=CANTIDAD_CLASES_PRINCIPALES_A_SELECCIONAR,
        return_all_samples_for_evaluation=True,
        semilla_aleatoria_split=SEMILLA_ALEATORIA
    )

    if dataset_completo_para_features is None or len(dataset_completo_para_features) == 0:
        print("No hay datos para extraer características para H2O. Saliendo.")
        h2o.cluster().shutdown()
        exit()

    cargador_caracteristicas = DataLoader(dataset_completo_para_features, batch_size=32, shuffle=False,
                                          num_workers=2 if os.name == 'nt' else 4, pin_memory=True,
                                          persistent_workers=True)

    extractor_caracteristicas = construir_extractor_caracteristicas(ruta_modelo_cnn,
                                                                    len(nombres_clases_globales),
                                                                    dispositivo_torch)

    caracteristicas_extraidas, etiquetas_extraidas = extraer_caracteristicas_desde_cargador(extractor_caracteristicas,
                                                                                               cargador_caracteristicas,
                                                                                               dispositivo_torch)

    if caracteristicas_extraidas.size == 0:
        print("No se extrajeron características para H2O. Saliendo.")
        h2o.cluster().shutdown()
        exit()

    # Dividir los datos extraídos en entrenamiento y prueba para H2O
    sss_h2o = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA_ALEATORIA)
    idx_train_h2o, idx_test_h2o = next(sss_h2o.split(caracteristicas_extraidas, etiquetas_extraidas))

    X_train_h2o, X_test_h2o = caracteristicas_extraidas[idx_train_h2o], caracteristicas_extraidas[idx_test_h2o]
    y_train_h2o, y_test_h2o = etiquetas_extraidas[idx_train_h2o], etiquetas_extraidas[idx_test_h2o]

    print(f"Dimensiones de los datos para H2O: Entrenamiento {X_train_h2o.shape}, Prueba {X_test_h2o.shape}")

    # Entrenar o cargar el modelo H2O
    if os.path.exists(ruta_modelo_h2o):
        print(f"Modelo H2O encontrado en {ruta_modelo_h2o}. Cargando modelo H2O.")
        modelo_h2o_final = h2o.load_model(ruta_modelo_h2o)
        print(f"Modelo H2O GBM cargado desde: {ruta_modelo_h2o}")
    else:
        print("Modelo H2O no encontrado. Iniciando entrenamiento de H2O GBM.")
        modelo_h2o_final = entrenar_h2o_gbm(X_train_h2o, y_train_h2o, X_test_h2o, y_test_h2o,
                                            nombres_clases_globales, ruta_modelo_h2o)

    # Evaluar el modelo H2O en el conjunto de prueba
    if modelo_h2o_final:
        evaluar_h2o_gbm(modelo_h2o_final, X_test_h2o, y_test_h2o, nombres_clases_globales)

    # Apagar el clúster H2O al finalizar
    h2o.cluster().shutdown()
    print("Script finalizado.")