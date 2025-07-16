from app import Pruebasdos
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles # type: ignore
from starlette.responses import FileResponse
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import os, logging, shutil
import pandas as pd
from pathlib import Path

from app.Servicio import Servicio

app = FastAPI()


# Configuración básica de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directorio donde se guardarán los archivos subidos
UPLOAD_DIRECTORY = "uploads"

# Creamos la instancia solo una vez (por ejemplo al arrancar)
servicio = Servicio(logger, 
                     nombre_modelo="LR_paso_cero.pkl",
                     carpeta_uploads=UPLOAD_DIRECTORY)



# Verificar si el directorio UPLOAD_DIRECTORY existe y crearlo si no existe
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    logger.info(f"Directorio '{UPLOAD_DIRECTORY}' creado correctamente")

# Montar el directorio "/static" como archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/modelos", StaticFiles(directory="modelos"), name="modelos")
app.mount("/resultados", StaticFiles(directory="resultados"), name="resultados")

templates = Jinja2Templates(directory="templates") 

RESULTADOS_DIR = Path("resultados").resolve()  

# Configurar CORS para permitir todas las solicitudes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir acceso desde todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados HTTP
)

# API para manejar la carga de archivos y guardarlos en el directorio UPLOAD_DIRECTORY
@app.post("/infiere") 
async def upload_file(file: UploadFile = File(...)):
    logger.info("Se ha accedido al endpoint /infiere de inferencia, borrando resultados anteriores")
    limpiar_resultados()

    # Construir la ruta completa del archivo
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

    # Guardar el archivo subido en el directorio de almacenamiento
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    #Creo la clase inferncias
    logger.info(f"🚶🏻‍♀️‍➡️----->/INFIERE - LEYENDO FICHERO {file.filename} y ")
    datos = servicio.leer_csv(file.filename)
    
    logger.info("🚶🏻‍♀️‍➡️----->/INFIERE - REALIZANDO  INFERENCIAS")
    predicciones = servicio.haz_inferencia(datos)
    
    logger.info("🚶🏻‍♀️‍➡️----->/INFIERE - GUARDANDO RESULTADOS")
    servicio.guardar_csv(predicciones, "Resultados de las inferencias.csv")
    
    
    if(predicciones.isnull):
        return {"mensaje": "no se ha podido predecir NISUPUTAMADRE"}
    else: 
         return {"mensaje": "parece que alguna predicción se ha hecho"}

# Ruta principal para servir el archivo index.html
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/resultados", response_class=HTMLResponse)
def list_files(request: Request):

    files = os.listdir("./resultados")
    files_paths = sorted([f"{request.url._url}/{f}" for f in files])
    print(files_paths)
    return templates.TemplateResponse(
        "list_files.html", {"request": request, "files": files_paths}
    )

def limpiar_resultados():
    """
    Borra todos los archivos y subcarpetas dentro de /resultados
    cada vez que se levanta la aplicación.
    """
    if not RESULTADOS_DIR.exists():
        RESULTADOS_DIR.mkdir()
        logger.info("Creada carpeta %s", RESULTADOS_DIR)
        return

    # recorrer y eliminar
    for p in RESULTADOS_DIR.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
            logger.info("Eliminado %s", p)
        except Exception as e:
            logger.error("No se pudo eliminar %s → %s", p, e)

    logger.info("📁 Carpeta resultados vaciada antes de realizar inferencias")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9000)

