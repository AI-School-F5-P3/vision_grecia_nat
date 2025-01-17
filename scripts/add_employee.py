import os
import cv2
import face_recognition
import pickle
from datetime import datetime
import click
import sys

# Añadir el directorio raíz al path para poder importar desde src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def capture_photos(name, num_photos=5):
    """Captura fotos del empleado usando la webcam"""
    config = Config()
    
    # Crear directorio para el empleado si no existe
    employee_dir = os.path.join(config.EMPLOYEES_DIR, name)
    os.makedirs(employee_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    photos_taken = 0
    print(f"\nCapturando {num_photos} fotos para {name}")
    print("Presiona ESPACIO para capturar una foto")
    print("Presiona ESC para cancelar")
    
    while photos_taken < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar video")
            break
            
        # Mostrar contador de fotos
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Fotos: {photos_taken}/{num_photos}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Detectar rostros en tiempo real
        face_locations = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for top, right, bottom, left in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("Captura de Fotos", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # ESPACIO
            if len(face_locations) == 1:  # Asegurar que solo hay una cara
                # Guardar la foto
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_path = os.path.join(employee_dir, f"{name}_{timestamp}.jpg")
                cv2.imwrite(photo_path, frame)
                photos_taken += 1
                print(f"Foto {photos_taken} capturada")
            else:
                print("¡Asegúrate de que solo hay una persona en el frame!")
    
    cap.release()
    cv2.destroyAllWindows()
    return photos_taken > 0

def generate_encodings():
    """Genera encodings para todas las fotos de empleados"""
    config = Config()
    
    print("\nGenerando encodings faciales...")
    known_encodings = []
    known_names = []
    
    # Recorrer directorio de empleados
    for employee_name in os.listdir(config.EMPLOYEES_DIR):
        employee_dir = os.path.join(config.EMPLOYEES_DIR, employee_name)
        if not os.path.isdir(employee_dir):
            continue
            
        print(f"Procesando fotos de {employee_name}")
        for photo_name in os.listdir(employee_dir):
            if not photo_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            photo_path = os.path.join(employee_dir, photo_name)
            image = face_recognition.load_image_file(photo_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(employee_name)
    
    # Guardar encodings
    print(f"\nGuardando {len(known_encodings)} encodings...")
    os.makedirs(os.path.dirname(config.ENCODINGS_FILE), exist_ok=True)
    with open(config.ENCODINGS_FILE, 'wb') as f:
        pickle.dump({
            'encodings': known_encodings,
            'names': known_names
        }, f)
    
    print("¡Encodings generados y guardados exitosamente!")
    return len(known_encodings)

@click.command()
@click.argument('name')
@click.option('--num-photos', default=5, help='Número de fotos a capturar')
def main(name, num_photos):
    """Script para añadir un nuevo empleado al sistema"""
    print(f"Añadiendo nuevo empleado: {name}")
    
    if capture_photos(name, num_photos):
        num_encodings = generate_encodings()
        print(f"\nProceso completado. Total de encodings: {num_encodings}")
    else:
        print("\nProceso cancelado")

if __name__ == '__main__':
    main()