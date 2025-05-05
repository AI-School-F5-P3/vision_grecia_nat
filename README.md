# Sistema de Reconocimiento Facial para Control de Acceso

Este proyecto implementa un sistema de control de acceso mediante reconocimiento facial, diseñado para registrar y verificar el ingreso de empleados utilizando visión por computadora.

## Propósito
Automatizar el registro de entradas y salidas de empleados mediante reconocimiento facial, mejorando la seguridad y facilitando la gestión de accesos.

## Requisitos
- Python 3.8+
- OpenCV
- face_recognition
- numpy
- click

Instala las dependencias ejecutando:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto
```
vision_grecia_nat/
├── .gitignore           # Archivos y carpetas ignorados por Git
├── README.md            # Documentación del proyecto
├── data/                # Datos (imágenes de empleados, encodings)
│   └── .gitkeep         # Marcador para mantener la carpeta en Git
├── main.py              # Script principal para iniciar el sistema
├── requirements.txt     # Dependencias de Python
├── scripts/             # Scripts auxiliares
│   ├── add_employee.py  # Script para registrar nuevos empleados
│   └── view_logs.py     # Script para visualizar registros de acceso
└── src/                 # Código fuente principal
    ├── __init__.py      # Inicializador del paquete src
    ├── config.py        # Configuraciones del sistema
    ├── logger.py        # Módulo para registrar eventos
    ├── recognition.py   # Lógica principal de reconocimiento facial
    └── utils.py         # Funciones de utilidad
```

## Uso Básico
### 1. Registrar empleados
Ejecuta el script correspondiente para capturar fotos de un nuevo empleado:
```bash
python scripts/add_employee.py --name "NombreEmpleado"
```
Sigue las instrucciones para capturar imágenes desde la cámara.

### 2. Generar encodings faciales
Esto se realiza automáticamente al registrar empleados, pero puedes forzarlo manualmente:
```bash
python scripts/generate_encodings.py
```

### 3. Iniciar el sistema de reconocimiento
```bash
python main.py
```
El sistema abrirá la cámara y mostrará los accesos permitidos o denegados en tiempo real.

### 4. Consultar registros de acceso
Puedes visualizar y analizar los registros ejecutando:
```bash
python scripts/view_logs.py
```

## ¿Cómo funciona el reconocimiento facial?
- El sistema utiliza la librería `face_recognition` para detectar y comparar rostros en tiempo real.
- Los encodings faciales de los empleados se almacenan y se usan para verificar la identidad al momento del acceso.
- Si el rostro coincide con un empleado registrado, el acceso es permitido y se registra el evento.

## Registro y gestión de empleados
- Las fotos de cada empleado se almacenan en la carpeta `data/empleados/`.
- Cada vez que se agrega un empleado, se generan nuevos encodings para mejorar la precisión.

## Consultar y exportar registros
- Los accesos se registran en la carpeta `logs/`.
- Puedes generar reportes en formato CSV o JSON usando la opción `--report` al ejecutar `main.py`.

## Contribuciones
¡Las contribuciones son bienvenidas! Por favor, abre un issue o pull request para sugerencias o mejoras.

## Licencia
Este proyecto es de uso académico y experimental.