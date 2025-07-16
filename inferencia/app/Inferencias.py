import logging
import joblib
import os
import configparser

class Inferencias:
    def __init__(self, logger: logging, modelo_id: int, base_path: str = "modelos"):
        """
        Inicializa la clase y carga un pipeline según el modelo_id.

        :param modelo_id: Entero que identifica qué pipeline cargar.
        :param base_path: Ruta base donde se almacenan los archivos .joblib.
        """
        self.modelo_id = modelo_id
        self.base_path = base_path
        
        logger.info('he llegado a crear la clase e inicializar el logger')
        #self.pipeline = self._cargar_pipeline()

    def _cargar_pipeline(self):
        """
        Carga el pipeline desde un archivo .joblib con base en modelo_id.
        """
        nombre_archivo = f"pipeline_{self.modelo_id}.joblib"
        ruta_completa = os.path.join(self.base_path, nombre_archivo)

        if not os.path.exists(ruta_completa):
            raise FileNotFoundError(f"No se encontró el archivo en cargar pipelineo: {ruta_completa}")

        return joblib.load(ruta_completa)

    def predecir(self, datos):
        """
        Usa el pipeline cargado para hacer predicciones.
        :param datos: Datos de entrada para el pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline no cargado correctamente.")
        return self.pipeline.predict(datos)
