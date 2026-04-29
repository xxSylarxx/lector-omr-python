import sys
import argparse
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path

# Agregar el directorio app al path para importar 
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as omr

app = Flask(__name__)

try:
    _plantilla = omr.cargar_plantilla()
except Exception as e:
    print(f"[ERROR] No se pudo cargar la plantilla: {e}")
    sys.exit(1)

EXTENSIONES_PERMITIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
TAMANO_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


def _procesar_archivo(archivo):
    """Procesa un FileStorage y devuelve un dict con el resultado o el error."""
    nombre = archivo.filename or "sin_nombre"

    ext = Path(nombre).suffix.lower()
    if ext not in EXTENSIONES_PERMITIDAS:
        return {
            "archivo": nombre,
            "ok": False,
            "error": f"Tipo de archivo no soportado: '{ext}'"
        }

    datos = archivo.read()
    if len(datos) > TAMANO_MAX_BYTES:
        return {"archivo": nombre, "ok": False, "error": "Imagen demasiado grande (max 20 MB)"}

    arr = np.frombuffer(datos, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"archivo": nombre, "ok": False, "error": "No se pudo decodificar la imagen"}

    try:
        respuestas = omr.process(img, _plantilla)
    except Exception as e:
        return {"archivo": nombre, "ok": False, "error": f"Error interno: {e}"}

    if not respuestas:
        return {"archivo": nombre, "ok": False, "error": "No se detecto un formulario valido"}

    return {
        "archivo": nombre,
        "ok": True,
        "total_preguntas": len(respuestas),
        "marcadas":  sum(1 for v in respuestas.values() if v is not None),
        "en_blanco": sum(1 for v in respuestas.values() if v is None),
        "respuestas": respuestas
    }


@app.post("/procesar")
def procesar():
    archivos = request.files.getlist("imagenes[]")
    if not archivos:
        archivo_unico = request.files.get("imagen")
        if archivo_unico:
            archivos = [archivo_unico]

    if not archivos:
        return jsonify({
            "ok": False,
            "error": "Falta el campo 'imagenes[]' (o 'imagen') en el formulario"
        }), 400

    resultados = [_procesar_archivo(a) for a in archivos]

    # La respuesta global es ok=true si al menos una imagen se proceso bien.
    hay_exito = any(r["ok"] for r in resultados)

    return jsonify({
        "ok": hay_exito,
        "total_imagenes": len(resultados),
        "resultados": resultados
    }), 200


@app.get("/ping")
def ping():
    """Verificar que el servidor esta vivo."""
    return jsonify({"ok": True, "mensaje": "OMR API activa"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor OMR API")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Puerto (default: 5000)")
    parser.add_argument("--debug", action="store_true",
                        help="Modo debug de Flask")
    args = parser.parse_args()

    print(f"[OMR API] Iniciando en http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
