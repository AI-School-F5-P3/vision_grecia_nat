import cv2
import sys
from src.config import Config
from src.recognition import load_encodings, recognize_faces
from src.utils import setup_signal_handler, draw_face_info, release_resources

def main():
    # Configurar manejador de señales
    setup_signal_handler()
    
    # Inicializar configuración
    config = Config()
    
    # Cargar encodings conocidos
    known_face_encodings, known_face_names = load_encodings(config.ENCODINGS_FILE)
    
    if not known_face_encodings:
        print("No hay empleados registrados en el sistema.")
        return
    
    # Inicializar la cámara
    print("Iniciando cámara...")
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    print("Sistema iniciado. Presiona 'q' para salir.")
    
    try:
        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el frame.")
                break
            
            # Reconocer rostros en el frame
            face_info = recognize_faces(
                frame, 
                known_face_encodings, 
                known_face_names, 
                tolerance=config.FACE_RECOGNITION_TOLERANCE
            )
            
            # Dibujar información en el frame
            frame = draw_face_info(frame, face_info)
            
            # Mostrar el frame
            cv2.imshow(config.WINDOW_NAME, frame)
            
            # Salir con 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Cerrando el programa...")
                break
                
    finally:
        # Liberar recursos
        release_resources(cap)
        print("Sistema finalizado.")
        sys.exit(0)

if __name__ == "__main__":
    main()