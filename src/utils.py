import cv2
import os
import sys
import signal
import logging
from datetime import datetime

def setup_logging(log_file=None, level=logging.INFO):
    """
    Configura el sistema de logging
    
    Args:
        log_file (str, optional): Ruta al archivo de log. Si es None, solo se muestra en consola.
        level (int): Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Formato del log
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configurar logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Salida a consola
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger()

def setup_signal_handler():
    """
    Configura el manejador de señales para cierre controlado
    """
    def signal_handler(sig, frame):
        print('\nPrograma terminado (Ctrl+C)')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

def draw_face_info(frame, face_info):
    """
    Dibuja información del rostro en el frame
    
    Args:
        frame (numpy.ndarray): Frame donde dibujar
        face_info (list): Lista de tuplas (nombre, coordenadas, color, texto_acceso)
        
    Returns:
        numpy.ndarray: Frame con la información dibujada
    """
    for name, (left, top, right, bottom), color, access_text in face_info:
        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Dibujar rectángulo para el nombre
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Mostrar nombre
        cv2.putText(frame, name, (left + 6, bottom - 6),
                  cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Mostrar estado de acceso
        cv2.putText(frame, access_text, (left, top - 10),
                  cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
    
    return frame

def ensure_dir_exists(directory):
    """
    Asegura que un directorio exista, creándolo si es necesario
    
    Args:
        directory (str): Ruta al directorio
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directorio creado: {directory}")

def format_timestamp(timestamp=None):
    """
    Formatea una marca de tiempo
    
    Args:
        timestamp (datetime, optional): Marca de tiempo a formatear. Si es None, se usa la hora actual.
        
    Returns:
        str: Marca de tiempo formateada
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_confidence_from_distance(face_distance):
    """
    Calcula el nivel de confianza a partir de la distancia facial
    
    Args:
        face_distance (float): Distancia facial
        
    Returns:
        float: Nivel de confianza (0.0 - 1.0)
    """
    if face_distance > 0.6:
        return 0.0
    return 1.0 - face_distance

def release_resources(cap=None):
    """
    Libera recursos como la cámara y ventanas de OpenCV
    
    Args:
        cap (cv2.VideoCapture, optional): Objeto de captura de video
    """
    print("\nLiberando recursos...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Necesario para asegurar que las ventanas se cierren


def handle_error(error, message=None, exit_code=None):
    """
    Maneja errores de forma centralizada
    
    Args:
        error (Exception): La excepción capturada
        message (str, optional): Mensaje personalizado para mostrar
        exit_code (int, optional): Código de salida si se debe terminar el programa
    """
    error_msg = message if message else str(error)
    logging.error(f"Error: {error_msg}")
    logging.debug(f"Detalles del error: {error}", exc_info=True)
    
    print(f"\n[ERROR] {error_msg}")
    
    if exit_code is not None:
        sys.exit(exit_code)

def validate_camera(camera_id):
    """
    Valida que la cámara esté disponible y funcionando
    
    Args:
        camera_id (int): ID de la cámara a validar
        
    Returns:
        bool: True si la cámara está disponible, False en caso contrario
    """
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except Exception:
        return False