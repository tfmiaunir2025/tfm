# app/Pruebas.py
import logging, joblib
import pandas as pd      
import numpy as np
import utils as ut
import joblib
import matplotlib.pyplot as plt
import shap
from shap import Explanation
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import FunctionTransformer



class Servicio:

    def __init__(self, logger: logging.Logger,
                 nombre_modelo: str = "LR_paso_cero.pkl",
                 carpeta_uploads: str = "uploads",
                 carpeta_resultados: str = "resultados"):
        self.logger = logger

        base_dir = Path(__file__).resolve().parent.parent   # …/inferencia/
        self.uploads_dir     = base_dir / carpeta_uploads
        self.resultados_dir  = base_dir / carpeta_resultados
        self.resultados_dir.mkdir(exist_ok=True)            # crea resultados/ si no existe
        self.nombre_modelo = nombre_modelo
        self.ruta_modelo = base_dir / "modelos" / nombre_modelo
        self.pipeline    = self._cargar_modelo()



    def haz_inferencia(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"SERVICIO.haz_inferencia 📄 ---> VOY A PROCESAR DATAFRAME")
        
        missing_cols = ut.expected_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f" SERVICIO.haz_inferencia 🚫 Faltan columnas en el CSV: {missing_cols}")
        else:
            self.logger.info(f"SERVICIO.haz_inferencia ✔️ Fichero tiene las columnas esperadas")

        if ut.EXPLICAR_COL in df.columns:
            indices_a_explicar = df[df[ut.EXPLICAR_COL] == 1].index.tolist()
            df = df.drop(columns=[ut.EXPLICAR_COL])
        else:
            indices_a_explicar = []

        self.logger.info(f"Hay que explicar instancias {indices_a_explicar}")   
        X = df.drop(columns=[ut.TARGET]) if ut.TARGET in df.columns else df

        self.logger.info(f"SERVICIO.haz_inferencia 🧙 invocando el modelo")
        preds = self.pipeline.predict(X)
        preds_df = pd.DataFrame({"Prediccion": preds})
        result_df = pd.concat([df, pd.DataFrame(preds_df, columns=['Prediccion'])], axis=1)
        
        #Explicar
        self.xai_shap(X, indices_a_explicar)
        if(len(indices_a_explicar)!=0):
            self.xai_lime(X, indices_a_explicar)
            
        self.logger.info(f"SERVICIO 🧙 voy a DEVOLVER predicciones")
        return result_df

    # ---------- SHAP ------------
    def xai_shap(self, X, necesitan_explicacion=None):
        
        X_transformed = self.pipeline.named_steps['preprocessor'].transform(X)   
        ohe =  self.pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_features = ohe.get_feature_names_out(ut.categoricas)
        feature_names = np.concatenate([cat_features, ut.binarias, ut.numericas]) 
        
        self.logger.info("SERVICIO.xai_shap 🔍 Generando explicabilidad global con SHAP agrupado")
        model_only = self.pipeline.named_steps['model']
        explainer_shap = shap.Explainer(model_only, X_transformed)
        shap_values = explainer_shap(X_transformed)

        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        X_df = pd.DataFrame(X_transformed, columns=feature_names)

        def agrupar_por_prefijo(columnas):
            return [col.split('_')[0] if '_' in col and col.split('_')[0] in ut.categoricas else col for col in columnas]

        agrupadas = agrupar_por_prefijo(feature_names)
        shap_df.columns = agrupadas
        X_df.columns = agrupadas

        shap_df_grouped = shap_df.groupby(axis=1, level=0).sum()
        X_df_grouped = X_df.groupby(axis=1, level=0).sum()
        # 1  Summary plot --------------------------
        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_df_grouped.values,
            features=X_df_grouped.values,
            feature_names=X_df_grouped.columns.tolist(),
            show=False
        )

        # 2⃣  Ruta de salida --------------------------
        agrupado = f"Importancia de las características según SHAP (desglose de valores).png"
        ruta_agupado_png   = self.resultados_dir / agrupado   # <- carpeta resultados/ del __init__

        # 3⃣  Guardar y cerrar ------------------------
        plt.savefig(ruta_agupado_png, bbox_inches="tight", dpi=150)
        plt.close()
        self.logger.info("SERVICIO.xai_shap 📈 SHAP summary plot agrupado guardado en %s", ruta_agupado_png)
        
        #4 Bar plot --------------------------
        shap_global = shap_df_grouped.mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        shap_global.plot(kind='bar')

        bar_plot = f"Importancia de las características según SHAP (valores totales).png"
        ruta_bar_plot   = self.resultados_dir / bar_plot
        plt.savefig(ruta_bar_plot, bbox_inches='tight')
        plt.close()
        self.logger.info("SERVICIO.xai_shap 📈 SHAP bar plot agrupado guardado en %s", ruta_bar_plot)

        #5 force plot --------------------------
        force_plot_html = shap.force_plot(
            base_value      =explainer_shap.expected_value,
            shap_values     =shap_df_grouped.values,
            features        =X_df_grouped.values,
            feature_names   =X_df_grouped.columns.tolist()
        )
        force_plot = f"shap_force_plot.html"
        ruta_force_plot   = self.resultados_dir / force_plot
        
        #shap.save_html(str(ruta_force_plot), force_plot_html)
        self.logger.info(f"SERVICIO.xai_shap ⚡ SHAP force plot global interactivo guardado en {ruta_force_plot}")

        #6 individuales shap ------------------------
        for idx in necesitan_explicacion:
            explanation_ind = Explanation(
                values=shap_df_grouped.iloc[idx].values,
                base_values=explainer_shap.expected_value,
                data=X_df_grouped.iloc[idx].values,
                feature_names=X_df_grouped.columns.tolist()
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation_ind, max_display=10, show=False)
            
            nombre_shap = f"Explicación SHAP para el registro {idx}.png"
            ruta_shap   = self.resultados_dir / nombre_shap   

            # Guardar y cerrar ------------------------
            plt.savefig(ruta_shap, bbox_inches="tight")
            plt.close()

            self.logger.info(f"SERVICIO.xai_shap 💧 SHAP individual guardado en {ruta_shap}")


    # ---------- LIME ------------
    def xai_lime(self, X, necesitan_explicacion): #necesita_explicacion = indices a Explicar

        self.logger.info(f"SERVICIO.xai_lime 👩🏻‍🏫🍋‍🟩 - Explicando Lime")

        X_transformed = self.pipeline.named_steps['preprocessor'].transform(X)   
        ohe =  self.pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_features = ohe.get_feature_names_out(ut.categoricas)
        feature_names = np.concatenate([cat_features, ut.binarias, ut.numericas]) 

        self.logger.info(f"SERVICIO.xai_lime procesado de datos a través del pipeline")

        explainer = LimeTabularExplainer(
            training_data=X_transformed,
            feature_names=feature_names,
            class_names=['No', 'Yes'],
            mode='classification'
        )

        self.logger.info(f"SERVICIO.xai_lime 👩🏻‍🏫🍋‍🟩 - Explicador LIME creado")

        for idx in necesitan_explicacion:
            exp = explainer.explain_instance(
                data_row=X_transformed[idx],
                predict_fn=self.predict_proba_lime,
                num_features=10
            )

            # ---------- construir ruta de salida ----------------------
            nombre_html = f"Explicación LIME para el registro {idx}.html"
            ruta_html   = self.resultados_dir / nombre_html
            self.logger.info(" SERVICIO.xai_lime 🍋‍🟩📄 GUARDAR LIME guardado en %s", ruta_html)
            exp.save_to_file(ruta_html)
            
            
    # === Función de predicción para LIME ===
    def predict_proba_lime(self, input_array):
        return self.pipeline.named_steps['model'].predict_proba(input_array)
    
        # ---------- guardar DataFrame en resultados ----------
    def guardar_csv(self, df: pd.DataFrame, nombre_salida: str) -> Path:
        ruta_out = self.resultados_dir / nombre_salida
        df.to_csv(ruta_out, index=False, encoding="utf-8")
        self.logger.info("SERVICIO.guardar_csv ✅ CSV escrito en %s (%d filas × %d columnas)",
                         ruta_out, df.shape[0], df.shape[1])
        return ruta_out


    # ---------- carga CSV ------------
    def leer_csv(self, nombre_csv: str) -> pd.DataFrame:
        """
        Lee un CSV que está dentro de la carpeta 'uploads' y lo devuelve como DataFrame
        """
        ruta_csv = self.uploads_dir / nombre_csv

        if not ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró {ruta_csv}")

        self.logger.info(f"SERVICIO.leer_csv 📄 Leyendo CSV {ruta_csv}")
        df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8")   # ajusta separador/encoding si hace falta
        self.logger.info(f"SERVICIO.leer_csv ✅ CSV cargado: {df.shape[0]} filas · {df.shape[1]} columnas")
        return df

    # ---------- carga modelo ----------
    def _cargar_modelo(self):
        self.logger.info(f"SERVICIO.cargar_modelo: 📦 Cargando modelo desde {self.ruta_modelo}")
        return joblib.load(self.ruta_modelo)
