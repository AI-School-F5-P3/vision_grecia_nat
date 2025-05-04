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
from src.recognition import capture_employee_photos, generate_encodings

@click.command()
@click.argument('name')
@click.option('--num-photos', default=5, help='Número de fotos a capturar')
def main(name, num_photos):
    """Script para añadir un nuevo empleado al sistema"""
    print(f"Añadiendo nuevo empleado: {name}")
    
    config = Config()
    
    if capture_employee_photos(
        name, 
        config.CAMERA_ID, 
        config.FRAME_WIDTH, 
        config.FRAME_HEIGHT, 
        config.EMPLOYEES_DIR, 
        num_photos
    ):
        num_encodings = generate_encodings(config.EMPLOYEES_DIR, config.ENCODINGS_FILE)
        print(f"\nProceso completado. Total de encodings: {num_encodings}")
    else:
        print("\nProceso cancelado")

if __name__ == '__main__':
    main()