import cv2
import sys
import os
import time
from src.config import Config
from src.recognition import load_encodings, recognize_faces
from src.utils import setup_signal_handler, draw_face_info, release_resources, validate_camera, setup_logging, handle_error

def main():
    # Configurar logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"vision_{time.strftime('%Y%m%d')}.log")
    logger = setup_logging(log_file)
    
    # Configurar manejador de señales
    setup_signal_handler()
    
    try:
        # Inicializar configuración
        config = Config()
        
        # Verificar directorios necesarios
        os.makedirs(config.EMPLOYEES_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.ENCODINGS_FILE), exist_ok=True)
        
        # Cargar encodings conocidos
        known_face_encodings, known_face_names = load_encodings(config.ENCODINGS_FILE)
        
        if not known_face_encodings:
            print("ADVERTENCIA: No hay empleados registrados en el sistema.")
            print("Utilice el script add_employee.py para añadir empleados.")
            print("¿Desea continuar de todos modos? (s/n)")
            response = input().lower()
            if response != 's' and response != 'si':
                print("Programa terminado.")
                return
        
        # Validar cámara
        if not validate_camera(config.CAMERA_ID):
            handle_error(
                Exception(f"No se pudo acceder a la cámara con ID {config.CAMERA_ID}"),
                "Verifique que la cámara esté conectada y no esté siendo utilizada por otra aplicación",
                exit_code=1
            )
        
        # Inicializar la cámara
        print("Iniciando cámara...")
        cap = cv2.VideoCapture(config.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        print("Sistema iniciado. Presiona 'q' para salir.")
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        try:
            while True:
                # Capturar frame
                ret, frame = cap.read()
                if not ret:
                    print("Error al capturar el frame. Reintentando...")
                    # Reintentar unas cuantas veces antes de rendirse
                    for _ in range(3):
                        time.sleep(0.5)
                        cap.release()
                        cap = cv2.VideoCapture(config.CAMERA_ID)
                        ret, frame = cap.read()
                        if ret:
                            break
                    
                    if not ret:
                        handle_error(
                            Exception("No se pudo recuperar la conexión con la cámara"),
                            exit_code=1
                        )
                
                # Calcular FPS
                frame_count += 1
                if frame_count >= 10:
                    end_time = time.time()
                    fps = frame_count / (end_time - fps_start_time)
                    frame_count = 0
                    fps_start_time = end_time
                
                # Reconocer rostros en el frame
                face_info = recognize_faces(
                    frame, 
                    known_face_encodings, 
                    known_face_names, 
                    tolerance=config.FACE_RECOGNITION_TOLERANCE
                )
                
                # Dibujar información en el frame
                frame = draw_face_info(frame, face_info)
                
                # Mostrar FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar el frame
                cv2.imshow(config.WINDOW_NAME, frame)
                
                # Salir con 'q'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Cerrando el programa...")
                    break
                    
        except KeyboardInterrupt:
            print("\nPrograma interrumpido por el usuario")
        except Exception as e:
            handle_error(e, "Error durante la ejecución del programa")
        finally:
            # Liberar recursos
            release_resources(cap)
            
    except Exception as e:
        handle_error(e, "Error al inicializar el programa", exit_code=1)
    
    print("Sistema finalizado.")
    sys.exit(0)

if __name__ == "__main__":
    main()