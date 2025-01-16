# Desconocido con más frames
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

class EnhancedFaceRecognition:
    def __init__(self):
        print("Iniciando sistema...")
        # Inicializamos detectores
        self.detector = MTCNN()
        self.facenet = FaceNet()
        
        # Inicializamos cámara
        print("Inicializando cámara...")
        self.camera = cv2.VideoCapture(0)
        
        # Verificar si la cámara se abrió correctamente
        if not self.camera.isOpened():
            raise Exception("Error: No se pudo acceder a la cámara")
            
        print("Cámara inicializada correctamente")
        
        # Configuramos la resolución de la cámara
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Resto de la inicialización
        self.similarity_threshold = 0.6
        self.temporal_window = 8
        self.min_consecutive_detections = 5
        self.detection_history = {}
        self.last_cleanup = datetime.now()
        self.face_features = {}
        
        
    
    

    
    def update_detection_history(self, face_id, name, confidence):
        """Actualiza el historial de detecciones temporales"""
        current_time = datetime.now()
        
        if face_id not in self.detection_history:
            self.detection_history[face_id] = deque(maxlen=self.temporal_window)
        
        self.detection_history[face_id].append({
            'name': name,
            'confidence': confidence,
            'timestamp': current_time
        })


    def get_temporal_consensus(self, face_id):
        """Obtiene el consenso temporal de identificación"""
        if face_id not in self.detection_history:
            return "Desconocido", 0.0
        
        history = self.detection_history[face_id]
        if len(history) < self.min_consecutive_detections:
            return "Desconocido", 0.0
        
        name_counts = {}
        total_confidence = {}
        
        for detection in history:
            name = detection['name']
            conf = detection['confidence']
            name_counts[name] = name_counts.get(name, 0) + 1
            total_confidence[name] = total_confidence.get(name, 0.0) + conf
        
        max_count = 0
        consensus_name = "Desconocido"
        consensus_confidence = 0.0
        
        for name, count in name_counts.items():
            if count > max_count:
                max_count = count
                consensus_name = name
                consensus_confidence = total_confidence[name] / count
        
        if max_count < self.min_consecutive_detections:
            return "Desconocido", 0.0
            
        return consensus_name, consensus_confidence



    def extract_face(self, image, face_data):
        try:
            x, y, w, h = face_data['box']
            x, y = abs(x), abs(y)
            
            # Agregamos un margen para capturar mejor el rostro
            margin = int(0.2 * w)  # 20% de margen
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
            face = image[y:y+h, x:x+w]
            
            if face.size == 0:
                return None
                
            # Mejoramos el preprocesamiento
            face = cv2.resize(face, (160, 160))
            face = face.astype('float32')
            
            # Normalización mejorada
            mean, std = face.mean(), face.std()
            face = (face - mean) / std  # Normalización más estándar para FaceNet
            
            return face
        except Exception as e:
            print(f"Error al extraer rostro: {e}")
            return None
        
    def load_known_faces(self, faces_dir):
        """Carga y procesa rostros conocidos"""
        print("Cargando rostros conocidos...")
        known_embeddings = []
        known_names = []
        
        # Verificar si el directorio existe
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            print(f"Creando directorio {faces_dir}")
            faces_path.mkdir(parents=True, exist_ok=True)
            print("Por favor, añade imágenes de rostros conocidos en la carpeta data/known_faces/")
            return [], []
            
        for face_path in faces_path.glob("*.*"):
            try:
                print(f"Procesando imagen: {face_path.name}")
                # Cargamos y procesamos imagen 
                image = cv2.imread(str(face_path))
                if image is None:
                    print(f"No se pudo cargar la imagen: {face_path}")
                    continue
                
                   
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Detectamos rostro
                faces = self.detector.detect_faces(image)
                
                if faces:
                    face = self.extract_face(image, faces[0])
                    if face is not None:
                        # Generamos embedding
                        embedding = self.facenet.embeddings(np.expand_dims(face, axis=0))[0]
                        
                    

                        known_embeddings.append(embedding)
                        known_names.append(face_path.stem)
                        print(f"Rostro cargado exitosamente: {face_path.stem}")
                    else:
                        print(f"No se pudo extraer el rostro de la imagen: {face_path}")
                else:
                    print(f"No se detectó rostro en: {face_path}")
            
            except Exception as e:
                print(f"Error procesando {face_path}: {e}")
                continue
        
        # Validaciones y visualización de datos cargados
        print(f"Total de rostros cargados: {len(known_names)}")
        for idx, name in enumerate(known_names):
            print(f"Nombre: {name}, Embedding shape: {known_embeddings[idx].shape}")

        if len(known_names) != len(known_embeddings):
            print("Advertencia: La cantidad de nombres y embeddings no coincide. Revisa las imágenes de entrada.")
        
        return known_embeddings, known_names

    
    def find_matches(self, embedding, known_embeddings, known_names):
        """Encuentra coincidencias usando distancia euclidiana"""
        matches = []
        for idx, known_embedding in enumerate(known_embeddings):
            distance = np.linalg.norm(embedding - known_embedding)
            similarity = 1 / (1 + distance)
            if similarity > self.similarity_threshold:
                matches.append((known_names[idx], similarity))
        return matches

    def process_frame(self, frame, known_embeddings, known_names):
            """Procesa cada frame para detectar y reconocer rostros"""
            try:
                # Convertimos a RGB para MTCNN
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detectamos rostros
                faces = self.detector.detect_faces(rgb_frame)
                
                results = []
                for face_data in faces:
                    if face_data['confidence'] < 0.90:
                        continue
                        
                    face = self.extract_face(rgb_frame, face_data)
                    if face is None:
                        continue
                    
                    # Generamos embedding
                    embedding = self.facenet.embeddings(np.expand_dims(face, axis=0))[0]
                    

                    
                    # Generamos ID único para este rostro
                    face_id = f"{face_data['box'][0]}_{face_data['box'][1]}"
                    
                    # Buscamos coincidencias
                    matches = self.find_matches(embedding, known_embeddings, known_names)
                    
                    if matches:
                        name, confidence = max(matches, key=lambda x: x[1])
                    else:
                        name, confidence = "Desconocido", 0.0
                    
                    # Actualizamos historial temporal
                    self.update_detection_history(face_id, name, confidence)
                    
                    # Obtenemos consenso temporal
                    consensus_name, consensus_confidence = self.get_temporal_consensus(face_id)
                    
                    results.append((face_data['box'], consensus_name, consensus_confidence))
                
                return results
                
            except Exception as e:
                print(f"Error procesando frame: {e}")
                return []




    def draw_results(self, frame, results):
            """Dibuja los resultados en el frame"""
            for (x, y, w, h), name, confidence in results:
                # Dibujamos bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Añadimos nombre y confianza
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return frame



    def run(self):
        """Ejecuta el sistema mejorado"""
        try:
            print("Iniciando sistema de reconocimiento facial...")
            known_embeddings, known_names = self.load_known_faces("data/known_faces")
            
            # Verificación de los datos cargados
            if not known_embeddings or not known_names:
                print("No se encontraron rostros conocidos. Por favor, añade imágenes a data/known_faces/")
                return
            
            if len(known_embeddings) != len(known_names):
                print("Error: La cantidad de embeddings y nombres no coincide. Revisa las imágenes de data/known_faces/")
                return
            
            print("Sistema iniciado. Presiona 'q' para salir.")
            
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error al leer frame de la cámara")
                    break
                
                # Verificar si el frame es válido
                if frame is None or frame.size == 0:
                    print("Frame inválido recibido de la cámara")
                    continue
                
                results = self.process_frame(frame, known_embeddings, known_names)
                frame = self.draw_results(frame, results)
                
                cv2.imshow('Reconocimiento Facial Mejorado', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"Error en la ejecución: {e}")
        finally:
            print("Cerrando sistema...")
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        system = EnhancedFaceRecognition()
        system.run()
    except Exception as e:
        print(f"Error al iniciar el sistema: {e}")