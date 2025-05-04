import cv2
import sys
import os
import time
import argparse
from src.config import Config
from src.recognition import load_encodings, recognize_faces
from src.utils import setup_signal_handler, draw_face_info, release_resources, validate_camera, setup_logging, handle_error
from src.logger import AccessLogger

def parse_arguments():
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Sistema de reconocimiento facial')
    parser.add_argument('--no-log', action='store_true', help='Desactiva el registro de accesos')
    parser.add_argument('--report', action='store_true', help='Genera un reporte de accesos al finalizar')
    parser.add_argument('--report-format', choices=['csv', 'json'], default='csv', help='Formato del reporte')
    return parser.parse_args()

def main():
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"vision_{time.strftime('%Y%m%d')}.log")
    logger = setup_logging(log_file)
    
    # Inicializar logger de accesos
    access_logger = None
    if not args.no_log:
        access_logger = AccessLogger(log_dir=log_dir)
        print(f"Registro de accesos activado. Logs en: {log_dir}")
    
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
                    tolerance=config.FACE_RECOGNITION_TOLERANCE,
                    access_logger=access_logger,
                    camera_id=config.CAMERA_ID
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
            # Generar reporte si se solicitó
            if args.report and access_logger:
                report_path = access_logger.generate_report(format_type=args.report_format)
                if report_path:
                    print(f"Reporte generado: {report_path}")
            
            # Liberar recursos
            release_resources(cap)
            
    except Exception as e:
        handle_error(e, "Error al inicializar el programa", exit_code=1)
    
    print("Sistema finalizado.")
    sys.exit(0)

if __name__ == "__main__":
    main()