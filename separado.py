import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from pathlib import Path

class PretrainedFaceRecognition:
    def __init__(self):
        # Inicializamos detectores
        self.detector = MTCNN()
        self.facenet = FaceNet()
        
        # Inicializamos cámara
        self.camera = cv2.VideoCapture(0)
        
        # Umbral de similitud
        self.similarity_threshold = 0.85
        
    def extract_face(self, image, face_data):
        """Extrae y preprocesa el rostro detectado"""
        x, y, w, h = face_data['box']
        x, y = abs(x), abs(y)
        face = image[y:y+h, x:x+w]
        
        # Redimensionar a 160x160 (requerido por FaceNet)
        face = cv2.resize(face, (160, 160))
        
        # Normalizar
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        
        return face

    def load_known_faces(self, faces_dir):
        """Carga y procesa rostros conocidos desde carpetas individuales"""
        known_embeddings = []
        known_names = []
        
        for person_dir in Path(faces_dir).iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            for face_path in person_dir.glob("*.*"):
                # Cargamos y procesamos imagen
                image = cv2.imread(str(face_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detectamos rostro
                faces = self.detector.detect_faces(image)
                if faces:
                    face = self.extract_face(image, faces[0])
                    # Generamos embedding
                    embedding = self.facenet.embeddings(np.expand_dims(face, axis=0))[0]
                    
                    known_embeddings.append(embedding)
                    known_names.append(person_name)
        
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
        # Convertimos a RGB para MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectamos rostros
        faces = self.detector.detect_faces(rgb_frame)
        
        results = []
        for face_data in faces:
            # Extraemos y procesamos rostro
            face = self.extract_face(rgb_frame, face_data)
            
            # Generamos embedding
            embedding = self.facenet.embeddings(np.expand_dims(face, axis=0))[0]
            
            # Buscamos coincidencias
            matches = self.find_matches(embedding, known_embeddings, known_names)
            
            if matches:
                name, confidence = max(matches, key=lambda x: x[1])
            else:
                name, confidence = "Desconocido", 0.0
                
            results.append((face_data['box'], name, confidence))
            
        return results

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
