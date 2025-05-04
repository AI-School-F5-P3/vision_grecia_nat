import os
import csv
import json
import logging
from datetime import datetime
import threading

class AccessLogger:
    """
    Clase para gestionar el registro de accesos al sistema
    """
    def __init__(self, log_dir="logs", csv_filename=None, json_filename=None):
        """
        Inicializa el logger de accesos
        
        Args:
            log_dir (str): Directorio donde se guardarán los logs
            csv_filename (str, optional): Nombre del archivo CSV para logs
            json_filename (str, optional): Nombre del archivo JSON para logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurar nombres de archivos
        date_str = datetime.now().strftime("%Y%m%d")
        self.csv_filename = csv_filename or f"accesos_{date_str}.csv"
        self.json_filename = json_filename or f"accesos_{date_str}.json"
        
        # Rutas completas
        self.csv_path = os.path.join(log_dir, self.csv_filename)
        self.json_path = os.path.join(log_dir, self.json_filename)
        
        # Inicializar archivos si no existen
        self._init_csv_file()
        self._init_json_file()
        
        # Mutex para acceso concurrente
        self.lock = threading.Lock()
        
        # Configurar logger
        self.logger = logging.getLogger("access_logger")
        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(log_dir, f"system_{date_str}.log"))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _init_csv_file(self):
        """Inicializa el archivo CSV si no existe"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'nombre', 'acceso', 'confianza', 'camara_id'])
    
    def _init_json_file(self):
        """Inicializa el archivo JSON si no existe"""
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump([], f)
    
    def log_access(self, name, access_granted, confidence=0.0, camera_id=0, extra_data=None):
        """
        Registra un acceso en los archivos de log
        
        Args:
            name (str): Nombre de la persona
            access_granted (bool): Si se concedió acceso o no
            confidence (float): Nivel de confianza del reconocimiento
            camera_id (int): ID de la cámara utilizada
            extra_data (dict, optional): Datos adicionales para incluir en el log
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Crear registro
        log_entry = {
            'timestamp': timestamp_str,
            'nombre': name,
            'acceso': 'PERMITIDO' if access_granted else 'DENEGADO',
            'confianza': round(confidence, 4),
            'camara_id': camera_id
        }
        
        # Añadir datos extra si existen
        if extra_data and isinstance(extra_data, dict):
            log_entry.update(extra_data)
        
        # Registrar en el logger de sistema
        access_status = "PERMITIDO" if access_granted else "DENEGADO"
        self.logger.info(f"Acceso {access_status} - Usuario: {name} - Confianza: {confidence:.4f}")
        
        # Usar lock para evitar problemas de concurrencia
        with self.lock:
            # Guardar en CSV
            try:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_str, 
                        name, 
                        'PERMITIDO' if access_granted else 'DENEGADO',
                        round(confidence, 4),
                        camera_id
                    ])
            except Exception as e:
                self.logger.error(f"Error al escribir en CSV: {e}")
            
            # Guardar en JSON
            try:
                # Leer datos existentes
                with open(self.json_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                
                # Añadir nuevo registro
                data.append(log_entry)
                
                # Guardar datos actualizados
                with open(self.json_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error al escribir en JSON: {e}")
    
    def get_access_history(self, name=None, start_date=None, end_date=None, access_type=None):
        """
        Obtiene el historial de accesos con filtros opcionales
        
        Args:
            name (str, optional): Filtrar por nombre
            start_date (datetime, optional): Fecha de inicio
            end_date (datetime, optional): Fecha de fin
            access_type (str, optional): Tipo de acceso ('PERMITIDO' o 'DENEGADO')
            
        Returns:
            list: Lista de registros que cumplen los criterios
        """
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            # Aplicar filtros
            filtered_data = data
            
            if name:
                filtered_data = [entry for entry in filtered_data if entry['nombre'] == name]
            
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
                start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
                filtered_data = [entry for entry in filtered_data if entry['timestamp'] >= start_str]
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
                end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
                filtered_data = [entry for entry in filtered_data if entry['timestamp'] <= end_str]
            
            if access_type:
                filtered_data = [entry for entry in filtered_data if entry['acceso'] == access_type]
            
            return filtered_data
        except Exception as e:
            self.logger.error(f"Error al obtener historial: {e}")
            return []
    
    def generate_report(self, output_file=None, format_type='csv'):
        """
        Genera un informe de accesos
        
        Args:
            output_file (str, optional): Ruta del archivo de salida
            format_type (str): Formato del informe ('csv' o 'json')
            
        Returns:
            str: Ruta del archivo generado
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"reporte_accesos_{timestamp}.{format_type}")
        
        try:
            data = self.get_access_history()
            
            if format_type.lower() == 'csv':
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Escribir encabezados
                    if data:
                        writer.writerow(data[0].keys())
                        # Escribir datos
                        for entry in data:
                            writer.writerow(entry.values())
            
            elif format_type.lower() == 'json':
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
            
            else:
                raise ValueError(f"Formato no soportado: {format_type}")
            
            self.logger.info(f"Reporte generado: {output_file}")
            return output_file
        
        except Exception as e:
            self.logger.error(f"Error al generar reporte: {e}")
            return None