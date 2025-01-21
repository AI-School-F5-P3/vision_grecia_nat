#El código de interfaz

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox,
                           QStackedWidget)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap,QFontDatabase
#Es QFontDatabase para mejorar el aspecto
import sys
import cv2
import face_recognition
import pickle
import os
from src.config import Config

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Sistema de Reconocimiento Facial')
        self.setGeometry(100, 100, 900, 700)
        
        # Load custom font
        QFontDatabase.addApplicationFont(":/resources/fonts/Roboto-Regular.ttf")        
        
        # Apply global stylesheet
        self.setStyleSheet('''
            QLabel {
                font-size: 18px;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 16px;
                padding: 10px;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:disabled {
                background-color: #CCC;
                color: #777;
            }
            QLineEdit {
                font-size: 16px;
                padding: 5px;
                border: 1px solid #CCC;
                border-radius: 5px;
            }
        ''')        
        
        
        
        
        
        
        # Widget central y layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Crear stack widget para múltiples pantallas
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        
        # Crear páginas
        self.main_page = self.create_main_page()
        self.register_page = self.create_register_page()
        self.recognition_page = self.create_recognition_page()
        
        # Agregar páginas al stack
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.register_page)
        self.stack.addWidget(self.recognition_page)
        
        # Variables para la cámara
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Cargar encodings
        self.load_encodings()
    
    def start_register(self):
        """Método para iniciar la página de registro"""
        self.stack.setCurrentIndex(1)
        self.start_camera()  # Iniciar la cámara al entrar en la página de registro
    
        
    def create_main_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


        
        # Título
        title = QLabel('Sistema de Reconocimiento Facial')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-size: 28px; font-weight: bold; margin-bottom: 30px;')
        layout.addWidget(title)
        
        # Botones
        register_btn = QPushButton('Registrar Nuevo Usuario')
        recognize_btn = QPushButton('Iniciar Reconocimiento')
        
        register_btn.clicked.connect(self.start_register)
        recognize_btn.clicked.connect(self.start_recognition)
        
        for btn in [register_btn, recognize_btn]:
            btn.setFixedWidth(300)
            layout.addWidget(btn)
        
        return page
    
    def create_register_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        
        # Título
        title = QLabel('Registro de Nuevo Usuario')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-size: 24px; font-weight: bold; margin-bottom: 20px;')
        layout.addWidget(title)
        
        # Campo de nombre
        name_layout = QHBoxLayout()
        name_label = QLabel('Nombre:')
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText('Ingrese su nombre aquí...')
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Vista previa de la cámara
        self.register_camera_label = QLabel('Vista previa de la cámara')
        self.register_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.register_camera_label.setStyleSheet('border: 1px solid #CCC; padding: 10px;')        
        layout.addWidget(self.register_camera_label)
        
        # Botones
        #                   tengo que borrar button_layout = QHBoxLayout()
        capture_btn = QPushButton('Capturar')
        back_btn = QPushButton('Volver')
        
        capture_btn.clicked.connect(self.capture_face)
        back_btn.clicked.connect(self.stop_camera)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(capture_btn)
        button_layout.addWidget(back_btn)
        layout.addLayout(button_layout)

        return page
    
    def create_recognition_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        
        # Vista Previa de la cámara
        self.recognition_camera_label = QLabel('Vista de la cámara')
        self.recognition_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recognition_camera_label.setStyleSheet('border: 1px solid #CCC; padding: 10px;')        
        layout.addWidget(self.recognition_camera_label)
        
        # Botón de volver
        back_btn = QPushButton('Detener y Volver')
        back_btn.clicked.connect(self.stop_recognition)
        layout.addWidget(back_btn)
        
        return page
    
    def load_encodings(self):
        self.known_encodings = []
        self.known_names = []
        if os.path.exists(self.config.ENCODINGS_FILE):
            with open(self.config.ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
    
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.config.CAMERA_ID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
            self.timer.start(30)
    
    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.stack.setCurrentIndex(0)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convertir frame a formato Qt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            
            # Mostrar frame en la etiqueta correspondiente
            if self.stack.currentIndex() == 1:
                self.register_camera_label.setPixmap(scaled_pixmap)
            else:
                # Reconocimiento facial

                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                # Dibujar resultados
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(
                        self.known_encodings, 
                        face_encoding,
                        tolerance=self.config.FACE_RECOGNITION_TOLERANCE
                    )
                    
                    name = "Desconocido"
                    color = (255, 0, 0)  # Rojo para desconocidos
                    access_text = "ACCESO DENEGADO"
                    
                    if True in matches:
                        match_index = matches.index(True)
                        name = self.known_names[match_index]
                        color = (0, 255, 0)  # Verde para conocidos
                        access_text = "ACCESO PERMITIDO"
                    
                    # Dibujar rectángulo y nombre
                    cv2.rectangle(rgb_image, (left, top), (right, bottom), color, 2)
                    
                    # Dibujar nombre
                    cv2.putText(rgb_image, name, (left + 6, bottom + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Dibujar estado de acceso
                    cv2.putText(rgb_image, access_text, (left, top - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)                
                
                # Convertir frame procesado a Qt
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.recognition_camera_label.setPixmap(scaled_pixmap)
    
    def capture_face(self):
        if not self.name_input.text():
            QMessageBox.warning(self, 'Error', 'Por favor ingrese un nombre')
            return
        
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) == 0:
                QMessageBox.warning(self, 'Error', 'No se detectó ningún rostro')
                return
            
            if len(face_locations) > 1:
                QMessageBox.warning(self, 'Error', 'Se detectó más de un rostro')
                return
            
            # Obtener encoding
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            
            # Guardar encoding
            self.known_encodings.append(face_encoding)
            self.known_names.append(self.name_input.text())
            
            data = {
                "encodings": self.known_encodings,
                "names": self.known_names
            }
            
            os.makedirs(os.path.dirname(self.config.ENCODINGS_FILE), exist_ok=True)
            with open(self.config.ENCODINGS_FILE, 'wb') as f:
                pickle.dump(data, f)
            
            QMessageBox.information(self, 'Éxito', 
                                  f'Usuario {self.name_input.text()} registrado exitosamente')
            
            # Limpiar y volver
            self.name_input.clear()
            self.stop_camera()
    
    def start_recognition(self):
        
        if not self.known_encodings:
            QMessageBox.warning(self, 'Error', 'No hay usuarios registrados')
            return
        
        self.stack.setCurrentIndex(2)
        self.start_camera()
    
    def stop_recognition(self):
        self.stop_camera()
    
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Establecer estilo global
    #app.setStyle('Fusion')
    
    # Crear y mostrar la ventana
    window = FaceRecognitionApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()