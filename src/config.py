import os

class Config:
    # Obtener el directorio base del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Configuraciones de video
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Rutas de archivos
    EMPLOYEES_DIR = os.path.join(BASE_DIR, "data", "empleados")
    ENCODINGS_FILE = os.path.join(BASE_DIR, "data", "encodings", "empleados_encodings.pkl")
    
    # Parámetros de reconocimiento facial
    FACE_RECOGNITION_TOLERANCE = 0.45  # Más estricto (valores más bajos = más estricto)
    MIN_FACE_SIZE = 20
    
    # Configuraciones de interfaz
    WINDOW_NAME = "Sistema de Acceso"
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2