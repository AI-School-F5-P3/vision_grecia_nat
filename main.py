import cv2
import face_recognition
import pickle
import os
import signal
import sys
from src.config import Config

def signal_handler(sig, frame):
    print('\nPrograma terminado (Ctrl+C)')
    sys.exit(0)

def load_encodings(config):
    """Carga los encodings conocidos desde el archivo"""
    print("Cargando encodings de empleados...")
    try:
        with open(config.ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f"Encodings cargados: {len(data['encodings'])} rostros")
        return data['encodings'], data['names']
    except FileNotFoundError:
        print("No se encontró el archivo de encodings")
        return [], []

def main():
    # Registrar manejador de señales para cierre controlado
    signal.signal(signal.SIGINT, signal_handler)
    
    # Inicializar configuración
    config = Config()
    
    # Cargar encodings conocidos
    known_face_encodings, known_face_names = load_encodings(config)
    
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
            
            # Redimensionar frame para procesamiento más rápido
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convertir de BGR (OpenCV) a RGB (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detectar rostros en el frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if face_locations:
                # Obtener encodings de los rostros detectados
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Ajustar coordenadas al tamaño original
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Buscar coincidencias
                    matches = face_recognition.compare_faces(
                        known_face_encodings, 
                        face_encoding,
                        tolerance=config.FACE_RECOGNITION_TOLERANCE
                    )
                    
                    name = "Desconocido"
                    
                    if True in matches:
                        # Calcular distancias faciales
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = matches.index(True)
                        
                        # Solo aceptar si la distancia es suficientemente pequeña
                        if face_distances[best_match_index] < config.FACE_RECOGNITION_TOLERANCE:
                            name = known_face_names[best_match_index]
                        else:
                            name = "Desconocido"  # Si la distancia es muy alta, marcar como desconocido
                    
                    # Configurar colores y mensaje de acceso
                    if name != "Desconocido":
                        color = (0, 255, 0)  # Verde para acceso permitido
                        access_text = "ACCESO PERMITIDO"
                    else:
                        color = (0, 0, 255)  # Rojo para acceso denegado
                        access_text = "ACCESO DENEGADO"
                    
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
            
            # Mostrar el frame
            cv2.imshow(config.WINDOW_NAME, frame)
            
            # Salir con 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Cerrando el programa...")
                break
                
    finally:
        print("\nLiberando recursos...")
        # Liberar recursos
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Necesario para asegurar que las ventanas se cierren
        print("Sistema finalizado.")
        sys.exit(0)

if __name__ == "__main__":
    main()