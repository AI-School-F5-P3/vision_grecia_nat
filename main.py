#Aquí va el archivo principal
from src.gui import FaceRecognitionApp
from PyQt6.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()