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

def get_timestamp():
    """
    Obtiene una marca de tiempo formateada
    
    Returns:
        str: Marca de tiempo en formato YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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