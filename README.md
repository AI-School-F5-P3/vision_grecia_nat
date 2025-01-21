#
# Face recognition app creada por el grupo vision_grecia_nat

## Descripción
Una aplicación de reconocimiento facial que utiliza procesamiento de imágenes y aprendizaje automático para identificar rostros en tiempo real. Este proyecto está diseñado para ser intuitivo y fácilmente integrable con otras soluciones.

---

## Características
- **Interfaz Gráfica (GUI):** Interactúa fácilmente con la aplicación mediante una interfaz amigable.
- **Reconocimiento Rápido:** Utiliza modelos eficientes para el reconocimiento facial.
- **Almacenamiento de Codificaciones:** Las codificaciones de rostros se guardan en un archivo `encodings.pickle` para su reutilización.
- **Configuración Personalizable:** Archivo `config.py` para ajustes específicos del usuario.

---

## Estructura del Proyecto
```plaintext
vision_grecia_nat/
│
├── encodings/
│   └── encodings.pickle       # Archivo con las codificaciones faciales
│
├── src/
│   ├── config.py              # Archivo de configuración
│   ├── gui.py                 # Código de la interfaz gráfica
│   ├── __init__.py            # Inicialización del módulo
│   └── __pycache__/           # Archivos cacheados de Python
│
├── .gitignore                 # Archivos ignorados por Git
├── main.py                    # Punto de entrada principal de la aplicación
├── README.md                  # Documentación del proyecto

```
---
## Archivos Clave
---

| Archivo                         | Descripción                                        |
|---------------------------------|----------------------------------------------------|
| **`main.py`**                   | Inicia la aplicación.                              |
| **`encodings/encodings.pickle`**| Contiene las codificaciones faciales guardadas.    |
| **`src/config.py`**             | Archivo de configuración con parámetros personalizables. |
| **`src/gui.py`**                | Código para la interfaz gráfica.                  |



## Bibliotecas Clave Utilizadas


| Biblioteca              | Descripción                                                                                  | Símbolo         |
|-------------------------|----------------------------------------------------------------------------------------------|-----------------|
| **`numpy`**             | Biblioteca para cálculo numérico y operaciones con arreglos multidimensionales.             | 🔢             |
| **`pandas`**            | Herramienta para manipulación y análisis de datos estructurados.                            | 📊             |
| **`tensorflow`**        | Framework para construir y entrenar modelos de aprendizaje automático y redes neuronales.    | 🤖             |
| **`scikit-learn`**      | Biblioteca para aprendizaje automático y minería de datos.                                   | 📚             |
| **`matplotlib`**        | Generación de gráficos en 2D para visualización de datos.                                    | 📈             |
| **`seaborn`**           | Biblioteca basada en Matplotlib para crear gráficos estadísticos atractivos.                 | 🌊             |
| **`flask`**             | Microframework para el desarrollo de aplicaciones web.                                       | 🌐             |
| **`fastapi`**           | Framework moderno y rápido para construir APIs.                                              | 🚀             |
| **`spacy`**             | Procesamiento de lenguaje natural avanzado.                                                 | 🧠             |
| **`transformers`**      | Herramientas para modelos de lenguaje natural como BERT y GPT.                               | 🗣️             |




## Instrucciones para Ejecutar la Aplicación

Sigue estos pasos para iniciar la aplicación y probar sus funcionalidades:

### Requisitos Previos
1. **Instala las Dependencias:**
   Asegúrate de haber instalado todas las librerías necesarias. Si no lo has hecho, ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

###
2. **Ejecuta la Aplicación: Utiliza el siguiente comando para iniciar el programa:**
  ```
  python main.py
  ```

