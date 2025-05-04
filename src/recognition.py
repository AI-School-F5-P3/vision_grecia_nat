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
        if not os.path.exists(encodings_file):
            print("No se encontró el archivo de encodings. Se creará uno nuevo cuando se registren empleados.")
            return [], []
            
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
            
        if not isinstance(data, dict) or 'encodings' not in data or 'names' not in data:
            raise ValueError("El archivo de encodings tiene un formato inválido")
            
        print(f"Encodings cargados: {len(data['encodings'])} rostros")
        return data['encodings'], data['names']
    except (FileNotFoundError, EOFError):
        print("No se encontró el archivo de encodings o está vacío")
        return [], []
    except (pickle.PickleError, ValueError) as e:
        print(f"Error al cargar encodings: {e}")
        print("Se creará un nuevo archivo de encodings")
        return [], []
    except Exception as e:
        print(f"Error inesperado al cargar encodings: {e}")
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
    
    if not os.path.exists(employees_dir):
        print(f"El directorio {employees_dir} no existe. Creándolo...")
        os.makedirs(employees_dir, exist_ok=True)
        return 0
        
    known_encodings = []
    known_names = []
    
    try:
        # Recorrer directorio de empleados
        employee_count = 0
        for employee_name in os.listdir(employees_dir):
            employee_dir = os.path.join(employees_dir, employee_name)
            if not os.path.isdir(employee_dir):
                continue
                
            print(f"Procesando fotos de {employee_name}")
            photo_count = 0
            
            for photo_name in os.listdir(employee_dir):
                if not photo_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                photo_path = os.path.join(employee_dir, photo_name)
                
                try:
                    image = face_recognition.load_image_file(photo_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(employee_name)
                        photo_count += 1
                    else:
                        print(f"  - No se detectó ningún rostro en {photo_name}")
                except Exception as e:
                    print(f"  - Error al procesar {photo_name}: {e}")
            
            if photo_count > 0:
                print(f"  - Se procesaron {photo_count} fotos con éxito")
                employee_count += 1
            else:
                print(f"  - No se pudo procesar ninguna foto para {employee_name}")
        
        if employee_count == 0:
            print("No se encontraron empleados con fotos válidas")
            return 0
    
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
    except Exception as e:
        print(f"Error al generar encodings: {e}")
        return 0

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
    if not name or name.strip() == "":
        print("Error: El nombre del empleado no puede estar vacío")
        return False
        
    # Validar caracteres no permitidos en nombres de archivo
    invalid_chars = '<>:"/\\|?*'
    if any(char in name for char in invalid_chars):
        print(f"Error: El nombre contiene caracteres no válidos: {invalid_chars}")
        return False
    
    # Crear directorio para el empleado si no existe
    employee_dir = os.path.join(employees_dir, name)
    try:
        os.makedirs(employee_dir, exist_ok=True)
    except Exception as e:
        print(f"Error al crear el directorio para {name}: {e}")
        return False
    
    # Inicializar cámara
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir la cámara con ID {camera_id}")
            return False
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return False
    
    photos_taken = 0
    print(f"\nCapturando {num_photos} fotos para {name}")
    print("Presiona ESPACIO para capturar una foto")
    print("Presiona ESC para cancelar")
    
    try:
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Mostrar rectángulos alrededor de los rostros
            for top, right, bottom, left in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Mostrar advertencia si se detectan múltiples rostros
            if len(face_locations) > 1:
                cv2.putText(display_frame, "¡MÚLTIPLES ROSTROS DETECTADOS!", 
                            (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 255), 2)
            
            cv2.imshow("Captura de Fotos", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Captura cancelada por el usuario")
                break
            elif key == 32:  # ESPACIO
                if len(face_locations) == 0:
                    print("¡No se detectó ningún rostro! Intenta de nuevo.")
                elif len(face_locations) > 1:
                    print("¡Se detectaron múltiples rostros! Asegúrate de que solo hay una persona en el frame.")
                else:
                    # Guardar la foto
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        photo_path = os.path.join(employee_dir, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(photo_path, frame)
                        photos_taken += 1
                        print(f"Foto {photos_taken} capturada")
                    except Exception as e:
                        print(f"Error al guardar la foto: {e}")
    except Exception as e:
        print(f"Error durante la captura: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Necesario para asegurar que las ventanas se cierren
    
    return photos_taken > 0

def recognize_faces(frame, known_face_encodings, known_face_names, tolerance=0.45, resize_factor=0.25, access_logger=None, camera_id=0):
    """
    Reconoce rostros en un frame y registra los accesos
    
    Args:
        frame (numpy.ndarray): Frame de video a analizar
        known_face_encodings (list): Lista de encodings conocidos
        known_face_names (list): Lista de nombres correspondientes a los encodings
        tolerance (float): Tolerancia para el reconocimiento facial (menor = más estricto)
        resize_factor (float): Factor para redimensionar el frame para procesamiento más rápido
        access_logger (AccessLogger, optional): Logger para registrar accesos
        camera_id (int): ID de la cámara utilizada
        
    Returns:
        list: Lista de tuplas (nombre, coordenadas, color, texto_acceso)
    """
    if frame is None:
        return []
        
    if not known_face_encodings or not known_face_names:
        return []
        
    results = []
    
    try:
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
                confidence = 0.0
                access_granted = False
                
                if True in matches:
                    # Calcular distancias faciales
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    # Solo aceptar si la distancia es suficientemente pequeña
                    if matches[best_match_index]:
                        confidence = 1.0 - face_distances[best_match_index]
                        if confidence >= (1.0 - tolerance):
                            name = known_face_names[best_match_index]
                            access_granted = True
                
                # Configurar colores y mensaje de acceso
                if access_granted:
                    color = (0, 255, 0)  # Verde para acceso permitido
                    access_text = f"ACCESO PERMITIDO ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Rojo para acceso denegado
                    access_text = "ACCESO DENEGADO"
                
                # Registrar acceso si hay un logger disponible
                if access_logger:
                    # Solo registrar si la confianza es suficiente o si es un desconocido
                    if access_granted or name == "Desconocido":
                        extra_data = {
                            'face_location': [top, right, bottom, left]
                        }
                        access_logger.log_access(
                            name=name,
                            access_granted=access_granted,
                            confidence=confidence,
                            camera_id=camera_id,
                            extra_data=extra_data
                        )
                
                results.append((name, (left, top, right, bottom), color, access_text))
    except Exception as e:
        print(f"Error en el reconocimiento facial: {e}")
    
    return results