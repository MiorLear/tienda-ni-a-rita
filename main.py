"""
Backend MVP - Tienda con audio
FastAPI + Groq (Whisper-large-v3 STT + Llama-3.1-8b LLM) + SQLite

Instalar:
  pip install fastapi uvicorn python-multipart groq sqlalchemy python-dotenv

Correr:
  uvicorn main:app --reload --port 8000

.env:
  GROQ_API_KEY=gsk_...
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from groq import Groq
import tempfile, os, json

app = FastAPI(title="Mi Tienda API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── BASE DE DATOS ───────────────────────────────────────────
DATABASE_URL = "sqlite:///./tienda.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Producto(Base):
    __tablename__ = "productos"
    id      = Column(Integer, primary_key=True, index=True)
    nombre  = Column(String, nullable=False)
    icono   = Column(String, default="paquete")
    stock   = Column(Integer, default=0)
    precio  = Column(Float, default=0.0)


class Venta(Base):
    __tablename__ = "ventas"
    id          = Column(Integer, primary_key=True, index=True)
    producto_id = Column(Integer)
    cantidad    = Column(Integer)
    total       = Column(Float)
    fecha       = Column(DateTime, default=datetime.now)


class EntradaInventario(Base):
    __tablename__ = "entradas"
    id          = Column(Integer, primary_key=True, index=True)
    producto_id = Column(Integer)
    cantidad    = Column(Integer)
    fecha       = Column(DateTime, default=datetime.now)


Base.metadata.create_all(bind=engine)


def seed_db():
    db = SessionLocal()
    if db.query(Producto).count() == 0:
        productos = [
            Producto(nombre="Shampoo",  icono="🧴", stock=12, precio=15.0),
            Producto(nombre="Jabon",    icono="🧼", stock=3,  precio=5.0),
            Producto(nombre="Refresco", icono="🥤", stock=24, precio=8.0),
            Producto(nombre="Arroz",    icono="🍚", stock=0,  precio=12.0),
            Producto(nombre="Aceite",   icono="🛢", stock=7,  precio=20.0),
            Producto(nombre="Dulces",   icono="🍬", stock=50, precio=1.0),
        ]
        db.add_all(productos)
        db.commit()
    db.close()


seed_db()


# ─── GROQ CLIENT ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "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"
)
client = Groq(api_key=GROQ_API_KEY)


# ─── STT: Whisper-large-v3 via Groq ──────────────────────────
def transcribir_audio(audio_bytes: bytes, filename: str = "recording.webm") -> str:
    """Transcribe audio con Whisper-large-v3 via Groq. Acepta webm, mp4, ogg, m4a."""
    # Determinar extension y mime type segun el archivo recibido
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
    mime_map = {
        "webm": "audio/webm",
        "mp4":  "audio/mp4",
        "m4a":  "audio/mp4",
        "ogg":  "audio/ogg",
        "wav":  "audio/wav",
        "flac": "audio/flac",
    }
    mime = mime_map.get(ext, "audio/webm")
    suffix = "." + ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resultado = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(filename, f, mime),
                language="es",
                response_format="text",
            )
        return resultado if isinstance(resultado, str) else resultado.text
    finally:
        os.unlink(tmp_path)


# ─── LLM: Llama-3.1-8b-instant via Groq ─────────────────────
def extraer_datos_audio(transcripcion: str, contexto: str, productos: list) -> dict:
    """Extrae producto y cantidad del texto transcrito usando Llama."""
    nombres = ", ".join([f"{p.id}:{p.nombre}" for p in productos])

    system_msg = (
        "Eres un asistente de tienda. "
        "Responde UNICAMENTE con JSON valido. Sin markdown, sin texto extra."
    )
    user_msg = (
        f'El usuario dijo: "{transcripcion}"\n\n'
        f"Contexto de la accion: {contexto}\n"
        "  - venta    = registrar una venta\n"
        "  - entrada  = agregar mercancia al inventario\n"
        "  - consulta = preguntar por el inventario\n\n"
        f"Productos disponibles (id:nombre): {nombres}\n\n"
        "Devuelve EXACTAMENTE este JSON:\n"
        "{\n"
        '  "product_id": <id numerico o null>,\n'
        '  "product_name": "<nombre o null>",\n'
        '  "quantity": <entero minimo 1>,\n'
        '  "intent": "<venta|entrada|consulta|unclear>",\n'
        '  "message": "<confirmacion breve y amigable en español>"\n'
        "}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=300,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Limpiar bloque ```json ... ``` si Llama lo incluye
    if "```" in raw:
        for parte in raw.split("```"):
            parte = parte.strip()
            if parte.startswith("json"):
                parte = parte[4:].strip()
            if parte.startswith("{"):
                raw = parte
                break

    return json.loads(raw)


# ─── MODELOS PYDANTIC ─────────────────────────────────────────
class VentaCreate(BaseModel):
    product_id: int
    quantity: int


class EntradaCreate(BaseModel):
    product_id: int
    quantity: int


# ─── ENDPOINTS ───────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "app": "Mi Tienda API",
        "stt": "groq/whisper-large-v3",
        "llm": "groq/llama-3.1-8b-instant",
    }


@app.get("/inventario")
def get_inventario():
    COLORS = ["#FFF0E8", "#F0FDF4", "#EFF6FF", "#FFFBEB", "#FDF4FF", "#FFF0E8"]
    db = SessionLocal()
    productos = db.query(Producto).all()
    db.close()
    return {
        "products": [
            {
                "id": p.id,
                "name": p.nombre,
                "icon": p.icono,
                "stock": p.stock,
                "price": p.precio,
                "color": COLORS[(p.id - 1) % len(COLORS)],
            }
            for p in productos
        ]
    }


@app.post("/ventas")
def registrar_venta(venta: VentaCreate):
    db = SessionLocal()
    producto = db.query(Producto).filter(Producto.id == venta.product_id).first()
    if not producto:
        db.close()
        raise HTTPException(404, "Producto no encontrado")
    if producto.stock < venta.quantity:
        db.close()
        raise HTTPException(400, f"Stock insuficiente. Solo hay {producto.stock} unidades.")

    producto.stock -= venta.quantity
    total = producto.precio * venta.quantity
    db.add(Venta(producto_id=venta.product_id, cantidad=venta.quantity, total=total))
    db.commit()
    stock_restante = producto.stock
    nombre = producto.nombre
    db.close()

    return {
        "success": True,
        "message": f"Venta registrada: {venta.quantity} {nombre} por Q{total:.2f}",
        "total": total,
        "remaining_stock": stock_restante,
    }


@app.post("/inventario/entrada")
def registrar_entrada(entrada: EntradaCreate):
    db = SessionLocal()
    producto = db.query(Producto).filter(Producto.id == entrada.product_id).first()
    if not producto:
        db.close()
        raise HTTPException(404, "Producto no encontrado")

    producto.stock += entrada.quantity
    db.add(EntradaInventario(producto_id=entrada.product_id, cantidad=entrada.quantity))
    db.commit()
    nuevo_stock = producto.stock
    nombre = producto.nombre
    db.close()

    return {
        "success": True,
        "message": f"Entrada registrada: {entrada.quantity} {nombre}. Total en bodega: {nuevo_stock}",
        "new_stock": nuevo_stock,
    }


@app.post("/audio/{contexto}")
async def procesar_audio(contexto: str, audio: UploadFile = File(...)):
    """
    1. Recibe audio (WebM/MP4/OGG)
    2. Transcribe con Whisper-large-v3 (Groq, gratis)
    3. Extrae intencion con Llama-3.1-8b-instant (Groq, gratis)
    """
    audio_bytes = await audio.read()
    filename = audio.filename or "recording.webm"

    # 1. Transcripcion
    try:
        transcripcion = transcribir_audio(audio_bytes, filename)
    except Exception as e:
        raise HTTPException(500, f"Error al transcribir audio: {str(e)}")

    # 2. Extraccion de intencion
    db = SessionLocal()
    productos = db.query(Producto).all()
    db.close()

    try:
        datos = extraer_datos_audio(transcripcion, contexto, productos)
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(500, f"Error al procesar con LLM: {str(e)}")

    datos["transcription"] = transcripcion
    datos["success"] = True

    # 3. Para consulta: adjuntar estado del inventario y resumen hablado
    if contexto == "consulta":
        db = SessionLocal()
        productos = db.query(Producto).all()
        db.close()
        sin_stock  = [p.nombre for p in productos if p.stock == 0]
        poco_stock = [p.nombre for p in productos if 0 < p.stock <= 5]
        partes = []
        if sin_stock:  partes.append(f"Sin stock: {', '.join(sin_stock)}")
        if poco_stock: partes.append(f"Poco stock: {', '.join(poco_stock)}")
        datos["message"]  = ". ".join(partes) if partes else "Todo el inventario esta bien."
        datos["products"] = [
            {"id": p.id, "name": p.nombre, "icon": p.icono, "stock": p.stock}
            for p in productos
        ]

    return datos


@app.get("/reportes/ventas-hoy")
def ventas_hoy():
    db = SessionLocal()
    hoy = datetime.now().date()
    ventas = db.query(Venta).filter(
        Venta.fecha >= datetime(hoy.year, hoy.month, hoy.day)
    ).all()
    total_monto = sum(v.total for v in ventas)
    db.close()
    return {
        "total_ventas": len(ventas),
        "total_monto": round(total_monto, 2),
        "fecha": str(hoy),
    }


# ─── WEBHOOK WHATSAPP (stub) ──────────────────────────────────
@app.post("/webhook/whatsapp")
async def whatsapp_webhook():
    """
    Conectar Twilio Sandbox o Meta Cloud API aqui.
    Ver README para instrucciones.
    """
    return {"status": "webhook activo"}


# ─── SERVIR FRONTEND (debe ir al final, despues de todas las rutas) ─────────
STATIC_DIR = Path(__file__).parent
if (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)