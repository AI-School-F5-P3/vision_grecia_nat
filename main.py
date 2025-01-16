

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from pathlib import Path



class PretrainedFaceRecognition:
    def __init__(self):
        self.detector = MTCNN()
        self.facenet = FaceNet()
        self.camera = cv2.VideoCapture(0)
        self.similarity_threshold = 0.65  # Reducido a 70%
        
        self.consecutive_frames = 0  # Contador para detecciones consecutivas
        self.last_detection = None  # Última detección confirmada
        self.min_consecutive = 3  # Mínimo de frames consecutivos para confirmar
        self.max_distance_threshold = 0.65  # Nuevo umbral de distancia máxima
    
    
    
    def extract_face(self, image, face_data):
        """Extrae y preprocesa el rostro detectado con manejo mejorado de dimensiones"""
        try:
            x, y, w, h = face_data['box']
            x, y = abs(x), abs(y)
            
            # Añadimos verificación de límites
            if x >= image.shape[1] or y >= image.shape[0]:
                return None
                
            # Añadimos margen para mejor detección
            margin_w = int(w * 0.2)
            margin_h = int(h * 0.2)
            
            # Ajustamos coordenadas con márgenes
            x = max(0, x - margin_w)
            y = max(0, y - margin_h)
            w = min(image.shape[1] - x, w + 2 * margin_w)
            h = min(image.shape[0] - y, h + 2 * margin_h)
            
            # Extraemos el rostro
            face = image[y:y+h, x:x+w]
            
            # Verificamos que el recorte fue exitoso
            if face.size == 0 or face is None:
                print("Error: Recorte de rostro fallido")
                return None
                
            # Redimensionamos a 160x160 (tamaño requerido por FaceNet)
            face = cv2.resize(face, (160, 160))
            
            # Verificamos la forma después del resize
            if face.shape != (160, 160, 3):
                print(f"Error: Dimensiones incorrectas después del resize: {face.shape}")
                return None
            
            # Convertimos a float32 y normalizamos
            face = face.astype('float32')
            face = (face - 127.5) / 128.0  # Normalización estándar para FaceNet
            
            # Verificación final de forma
            if face.shape != (160, 160, 3):
                print(f"Error: Dimensiones finales incorrectas: {face.shape}")
                return None
                
            return face
            
        except Exception as e:
            print(f"Error en extract_face: {e}")
            return None

    def load_known_faces(self, faces_dir):
        """Carga y procesa rostros conocidos"""
        known_embeddings = []
        known_names = []
        base_dir = Path(faces_dir)
        
        # Iteramos sobre cada carpeta de persona
        for person_dir in base_dir.iterdir():
            if person_dir.is_dir():  # Solo procesamos directorios
                person_name = person_dir.name
                print(f"Procesando imágenes de: {person_name}")
                
                # Procesamos cada imagen en el directorio de la persona
                for face_path in person_dir.glob("*.*"):
                    try:
                        print(f"  Procesando imagen: {face_path.name}")
                        image = cv2.imread(str(face_path))
                        if image is None:
                            print(f"  No se pudo cargar la imagen: {face_path}")
                            continue
                            
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = self.detector.detect_faces(image)
                        
                        if faces:
                            face = self.extract_face(image, faces[0])
                            if face is not None:
                                embedding = self.facenet.embeddings(np.expand_dims(face, axis=0))[0]
                                known_embeddings.append(embedding)
                                known_names.append(person_name)  # Usamos el nombre de la carpeta
                                print(f"  ✓ Rostro procesado exitosamente")
                        else:
                            print(f"  ✗ No se detectó rostro en: {face_path}")
                    
                    except Exception as e:
                        print(f"  ✗ Error procesando {face_path}: {e}")
                        continue
        
        print(f"\nTotal de rostros cargados: {len(known_names)}")
        # Imprimir resumen por persona
        name_counts = {}
        for name in known_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        print("\nResumen por persona:")
        for name, count in name_counts.items():
            print(f"- {name}: {count} imágenes procesadas")
        
        return known_embeddings, known_names
        
    
    
    
    
         
    def find_matches(self, embedding, known_embeddings, known_names):
        """Encuentra coincidencias usando distancia euclidiana"""
        matches = []
        for idx, known_embedding in enumerate(known_embeddings):
            distance = np.linalg.norm(embedding - known_embedding)
            similarity = 1 / (1 + distance)
            print(f"Comparando con {known_names[idx]}: Similaridad={similarity:.2f}, Distancia={distance:.2f}")
            if similarity > self.similarity_threshold:
                matches.append((known_names[idx], similarity))
        
        if not matches:
            print("No se encontraron coincidencias.")
        return matches



    def process_frame(self, frame, known_embeddings, known_names):
        
        """Procesamiento de frame con verificación temporal más estricta"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb_frame)
            
            results = []
            for face_data in faces:
                # Aumentamos el umbral de confianza de detección
                if face_data['confidence'] < 0.80:
                    continue
                    
                face = self.extract_face(rgb_frame, face_data)
                if face is None or face.shape != (160, 160, 3):
                    continue
                    
                face_array = np.expand_dims(face, axis=0)
                embedding = self.facenet.embeddings(face_array)[0]
                matches = self.find_matches(embedding, known_embeddings, known_names)
                
                if matches:
                    name, confidence = max(matches, key=lambda x: x[1])
                    
                    # Sistema de verificación temporal más estricto
                    if self.last_detection == name:
                        self.consecutive_frames += 1
                    else:
                        self.consecutive_frames = 1
                        
                    self.last_detection = name
                    
                    # Requiere más frames consecutivos y mayor confianza
                    if self.consecutive_frames >= self.min_consecutive and confidence > 0.85:
                        results.append((face_data['box'], name, confidence))
                    else:
                        results.append((face_data['box'], "Verificando...", confidence))
                else:
                    self.consecutive_frames = 0
                    self.last_detection = None
                    results.append((face_data['box'], "Desconocido", 0.0))
                    
            return results
            
        except Exception as e:
            print(f"Error en process_frame: {e}")
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
        """Ejecuta el sistema de reconocimiento facial"""
        # Cargamos rostros conocidos
        known_embeddings, known_names = self.load_known_faces("data/known_faces")
        
        print("Sistema iniciado. Presiona 'q' para salir.")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Procesamos frame
            results = self.process_frame(frame, known_embeddings, known_names)
            
            # Dibujamos resultados
            frame = self.draw_results(frame, results)
            
            # Mostramos resultado
            cv2.imshow('Reconocimiento Facial', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    system = PretrainedFaceRecognition()
    system.run()