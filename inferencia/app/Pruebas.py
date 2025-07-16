# app/Pruebas.py
from pathlib import Path
import utils
import joblib
import logging

class Pruebas:
    
    def __init__(self, logger: logging.Logger, nombre_modelo: str = "LR_paso_cero.pkl"):
        """
        nombre_modelo: nombre del archivo dentro de la carpeta modelos/
        """
        self.logger = logger

        # 1⃣ Carpeta raíz del proyecto  (= donde está este .py)
        base_dir = Path(__file__).resolve().parent.parent      #   .../app/  -> subimos 1 nivel (..)
        # 2⃣ Carpeta donde viven los modelos
        modelos_dir = base_dir / "modelos"                     #   .../modelos/
        # 3⃣ Ruta completa al modelo
        self.ruta_modelo = modelos_dir / nombre_modelo
        
        # Cargar el modelo
        self.pipeline = self._cargar_modelo()

    def _cargar_modelo(self):
        if not self.ruta_modelo.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {self.ruta_modelo!s}")
        
        self.logger.info(f"📦 Cargando modelo desde {self.ruta_modelo}")
        modelo = joblib.load(self.ruta_modelo)
        self.logger.info("✅ Modelo cargado correctamente")
        return modelo

    def metodo_prueba(self):
        self.logger.info("Pruebas.py -- Ejecutando método de prueba")

   





    
