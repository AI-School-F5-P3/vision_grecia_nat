# main.py
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from datetime import datetime
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer

class FaceRecognitionSystem:
    def __init__(self):
        # Inicializamos el detector y reconocedor facial
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        
        # Cargamos la cámara
        self.camera = cv2.VideoCapture(0)
        
        # Configuramos parámetros para optimizar velocidad
        self.frame_reduction = 0.25  # Reducimos tamaño del frame para procesamiento más rápido
        
    def load_known_faces(self, faces_dir):
        """Carga las imágenes de rostros conocidos"""
        known_faces = []
        known_names = []
        
        for face_path in Path(faces_dir).glob("*.*"):
            # Cargamos imagen y obtenemos encodings
            face_image = face_recognition.load_image_file(str(face_path))
            face_encoding = face_recognition.face_encodings(face_image)[0]
            
            known_faces.append(face_encoding)
            known_names.append(face_path.stem)  # Usamos nombre del archivo como nombre de persona
            
        return known_faces, known_names
        
    def process_frame(self, frame, known_faces, known_names):
        """Procesa cada frame para detectar y reconocer rostros"""
        # Reducimos tamaño del frame para mejor rendimiento
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_reduction, fy=self.frame_reduction)
        
        # Convertimos de BGR (OpenCV) a RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detectamos rostros en el frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # Comparamos con rostros conocidos
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            name = "Desconocido"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            
            face_names.append(name)
        
        return face_locations, face_names
    
    def draw_results(self, frame, face_locations, face_names):
        """Dibuja los resultados en el frame"""
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Reescalamos coordenadas al tamaño original
            top = int(top / self.frame_reduction)
            right = int(right / self.frame_reduction)
            bottom = int(bottom / self.frame_reduction)
            left = int(left / self.frame_reduction)
            
            # Dibujamos bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Dibujamos etiqueta con nombre
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Ejecuta el sistema de reconocimiento facial"""
        # Cargamos rostros conocidos
        known_faces, known_names = self.load_known_faces("data/known_faces")
        
        print("Sistema iniciado. Presiona 'q' para salir.")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Procesamos el frame
            face_locations, face_names = self.process_frame(frame, known_faces, known_names)
            
            # Dibujamos resultados
            frame = self.draw_results(frame, face_locations, face_names)
            
            # Mostramos resultado
            cv2.imshow('Reconocimiento Facial', frame)
            
            # Salimos si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()