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
from src.utils import validate_camera, handle_error

@click.command()
@click.argument('name')
@click.option('--num-photos', default=5, help='Número de fotos a capturar')
def main(name, num_photos):
    """Script para añadir un nuevo empleado al sistema"""
    try:
        print(f"Añadiendo nuevo empleado: {name}")
        
        # Validar nombre
        if not name or name.strip() == "":
            handle_error(ValueError("El nombre del empleado no puede estar vacío"), exit_code=1)
            
        # Validar número de fotos
        if num_photos < 1:
            handle_error(ValueError("El número de fotos debe ser al menos 1"), exit_code=1)
        
        config = Config()
        
        # Verificar directorios necesarios
        os.makedirs(config.EMPLOYEES_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.ENCODINGS_FILE), exist_ok=True)
        
        # Validar cámara
        if not validate_camera(config.CAMERA_ID):
            handle_error(
                Exception(f"No se pudo acceder a la cámara con ID {config.CAMERA_ID}"),
                "Verifique que la cámara esté conectada y no esté siendo utilizada por otra aplicación",
                exit_code=1
            )
        
        # Capturar fotos
        if capture_employee_photos(
            name, 
            config.CAMERA_ID, 
            config.FRAME_WIDTH, 
            config.FRAME_HEIGHT, 
            config.EMPLOYEES_DIR, 
            num_photos
        ):
            # Generar encodings
            num_encodings = generate_encodings(config.EMPLOYEES_DIR, config.ENCODINGS_FILE)
            
            if num_encodings > 0:
                print(f"\nProceso completado. Total de encodings: {num_encodings}")
            else:
                print("\nNo se pudieron generar encodings. Verifique las fotos capturadas.")
        else:
            print("\nProceso cancelado o no se pudieron capturar fotos")
            
    except Exception as e:
        handle_error(e, "Error al añadir empleado", exit_code=1)

if __name__ == '__main__':
    main()