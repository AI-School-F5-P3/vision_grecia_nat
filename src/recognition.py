import cv2
import face_recognition
import pickle
import os
import numpy as np
from datetime import datetime

def load_encodings(encodings_file):
    """
    Carga los encodings conocidos desde el archivo
    
    Args:
        encodings_file (str): Ruta al archivo de encodings
        
    Returns:
        tuple: (encodings, nombres) de los rostros conocidos
    """
    print("Cargando encodings de empleados...")
    try:
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Encodings cargados: {len(data['encodings'])} rostros")
        return data['encodings'], data['names']
    except FileNotFoundError:
        print("No se encontró el archivo de encodings")
        return [], []

def generate_encodings(employees_dir, encodings_file):
    """
    Genera encodings para todas las fotos de empleados
    
    Args:
        employees_dir (str): Directorio donde se almacenan las fotos de empleados
        encodings_file (str): Ruta donde se guardará el archivo de encodings
        
    Returns:
        int: Número de encodings generados
    """
    print("\nGenerando encodings faciales...")
    known_encodings = []
    known_names = []
    
    # Recorrer directorio de empleados
    for employee_name in os.listdir(employees_dir):
        employee_dir = os.path.join(employees_dir, employee_name)
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
    os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
    with open(encodings_file, 'wb') as f:
        pickle.dump({
            'encodings': known_encodings,
            'names': known_names
        }, f)
    
    print("¡Encodings generados y guardados exitosamente!")
    return len(known_encodings)

def capture_employee_photos(name, camera_id, frame_width, frame_height, employees_dir, num_photos=5):
    """
    Captura fotos del empleado usando la webcam
    
    Args:
        name (str): Nombre del empleado
        camera_id (int): ID de la cámara a utilizar
        frame_width (int): Ancho del frame
        frame_height (int): Alto del frame
        employees_dir (str): Directorio donde se guardarán las fotos
        num_photos (int): Número de fotos a capturar
        
    Returns:
        bool: True si se capturaron fotos, False en caso contrario
    """
    # Crear directorio para el empleado si no existe
    employee_dir = os.path.join(employees_dir, name)
    os.makedirs(employee_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
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

def recognize_faces(frame, known_face_encodings, known_face_names, tolerance=0.45, resize_factor=0.25):
    """
    Reconoce rostros en un frame
    
    Args:
        frame (numpy.ndarray): Frame de video a analizar
        known_face_encodings (list): Lista de encodings conocidos
        known_face_names (list): Lista de nombres correspondientes a los encodings
        tolerance (float): Tolerancia para el reconocimiento facial (menor = más estricto)
        resize_factor (float): Factor para redimensionar el frame para procesamiento más rápido
        
    Returns:
        list: Lista de tuplas (nombre, coordenadas, color, texto_acceso)
    """
    results = []
    
    # Redimensionar frame para procesamiento más rápido
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    
    # Convertir de BGR (OpenCV) a RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detectar rostros en el frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    if face_locations:
        # Obtener encodings de los rostros detectados
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Ajustar coordenadas al tamaño original
            top = int(top / resize_factor)
            right = int(right / resize_factor)
            bottom = int(bottom / resize_factor)
            left = int(left / resize_factor)
            
            # Buscar coincidencias
            matches = face_recognition.compare_faces(
                known_face_encodings, 
                face_encoding,
                tolerance=tolerance
            )
            
            name = "Desconocido"
            
            if True in matches:
                # Calcular distancias faciales
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = matches.index(True)
                
                # Solo aceptar si la distancia es suficientemente pequeña
                if face_distances[best_match_index] < tolerance:
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
            
            results.append((name, (left, top, right, bottom), color, access_text))
    
    return results