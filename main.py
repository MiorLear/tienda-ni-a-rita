"""
Backend MVP - Mi Tienda
FastAPI + Groq (Whisper STT + Llama LLM) + SQLite + ReportLab PDF

Instalar:
  pip install fastapi uvicorn python-multipart groq sqlalchemy python-dotenv reportlab aiofiles

Correr:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os, json, io, tempfile, calendar
from datetime import datetime
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, extract
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from groq import Groq

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, HRFlowable
)

# ─── CONFIG ───────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tienda.db")
APP_HOST     = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT     = int(os.getenv("APP_PORT", "8000"))
MONEDA       = os.getenv("MONEDA", "Q")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Falta GROQ_API_KEY en el archivo .env")

# ─── APP ──────────────────────────────────────────────────────
app = FastAPI(title="Mi Tienda API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── BASE DE DATOS ────────────────────────────────────────────
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()


class Producto(Base):
    __tablename__ = "productos"
    id      = Column(Integer, primary_key=True, index=True)
    nombre  = Column(String,  nullable=False)
    icono   = Column(String,  default="paquete")
    stock   = Column(Integer, default=0)
    precio  = Column(Float,   default=0.0)


class Venta(Base):
    __tablename__ = "ventas"
    id          = Column(Integer,  primary_key=True, index=True)
    producto_id = Column(Integer)
    cantidad    = Column(Integer)
    total       = Column(Float)
    fecha       = Column(DateTime, default=datetime.now)


class EntradaInventario(Base):
    __tablename__ = "entradas"
    id          = Column(Integer,  primary_key=True, index=True)
    producto_id = Column(Integer)
    cantidad    = Column(Integer)
    fecha       = Column(DateTime, default=datetime.now)


Base.metadata.create_all(bind=engine)


def seed_db():
    db = SessionLocal()
    if db.query(Producto).count() == 0:
        db.add_all([
            Producto(nombre="Shampoo",  icono="Shampoo",  stock=12, precio=15.0),
            Producto(nombre="Jabon",    icono="Jabon",    stock=3,  precio=5.0),
            Producto(nombre="Refresco", icono="Refresco", stock=24, precio=8.0),
            Producto(nombre="Arroz",    icono="Arroz",    stock=0,  precio=12.0),
            Producto(nombre="Aceite",   icono="Aceite",   stock=7,  precio=20.0),
            Producto(nombre="Dulces",   icono="Dulces",   stock=50, precio=1.0),
        ])
        db.commit()
    db.close()


seed_db()

# ─── GROQ ─────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)


def transcribir_audio(audio_bytes: bytes, filename: str = "recording.webm") -> str:
    """Transcribe audio con Whisper-large-v3 via Groq. Acepta webm, mp4, ogg, m4a."""
    ext      = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
    mime_map = {
        "webm": "audio/webm",
        "mp4":  "audio/mp4",
        "m4a":  "audio/mp4",
        "ogg":  "audio/ogg",
        "wav":  "audio/wav",
        "flac": "audio/flac",
    }
    mime   = mime_map.get(ext, "audio/webm")
    suffix = "." + ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resultado = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(filename, f, mime),
                language="es",
                response_format="text",
            )
        print(resultado)
        return resultado if isinstance(resultado, str) else resultado.text
    finally:
        os.unlink(tmp_path)


def extraer_datos_audio(transcripcion: str, contexto: str, productos: list) -> dict:
    # for producto in productos:
    #     print(producto.id, producto.nombre, producto.stock)
    # print(contexto)
    # print(transcripcion)
    """Extrae producto y cantidad del texto transcrito usando Llama-3.1-8b."""
    nombres    = ", ".join([f"{p.id}:{p.nombre}" for p in productos])
    system_msg = "Eres asistente de tienda. Responde SOLO con JSON valido. Sin markdown ni texto extra."
    user_msg   = (
        f'El usuario dijo: "{transcripcion}"\n\n'
        f"Contexto: {contexto} (venta=registrar venta, entrada=agregar inventario, consulta=preguntar inventario)\n"
        f"Productos (id:nombre): {nombres}\n\n"
        "Devuelve exactamente:\n"
        '{"product_id":<id o null>,"product_name":"<nombre o null>","quantity":<entero min 1>,'
        '"intent":"<venta|entrada|consulta|unclear>","message":"<confirmacion breve en español>"}'
    )
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=300,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        for parte in raw.split("```"):
            parte = parte.strip()
            if parte.startswith("json"):
                parte = parte[4:].strip()
            if parte.startswith("{"):
                raw = parte
                break
    print(raw)
    return json.loads(raw)


# ─── MODELOS ──────────────────────────────────────────────────
class VentaCreate(BaseModel):
    product_id: int
    quantity:   int


class EntradaCreate(BaseModel):
    product_id: int
    quantity:   int


# ─── HELPERS ──────────────────────────────────────────────────
COLORS_MAP = ["#FFF0E8", "#F0FDF4", "#EFF6FF", "#FFFBEB", "#FDF4FF", "#FFF0E8"]
ICONS_MAP  = {"Shampoo":"🧴","Jabon":"🧼","Refresco":"🥤","Arroz":"🍚","Aceite":"🛢","Dulces":"🍬"}


def producto_to_dict(p: Producto) -> dict:
    return {
        "id":    p.id,
        "name":  p.nombre,
        "icon":  ICONS_MAP.get(p.icono, "📦"),
        "stock": p.stock,
        "price": p.precio,
        "color": COLORS_MAP[(p.id - 1) % len(COLORS_MAP)],
    }


# ─── ENDPOINTS ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "stt": "groq/whisper-large-v3", "llm": "groq/llama-3.1-8b-instant"}


@app.get("/inventario")
def get_inventario():
    db = SessionLocal()
    productos = db.query(Producto).all()
    db.close()    
    return {"products": [producto_to_dict(p) for p in productos]}


@app.post("/ventas")
def registrar_venta(venta: VentaCreate):
    db = SessionLocal()
    p  = db.query(Producto).filter(Producto.id == venta.product_id).first()
    if not p:
        db.close()
        raise HTTPException(404, "Producto no encontrado")
    if p.stock < venta.quantity:
        db.close()
        raise HTTPException(400, f"Stock insuficiente. Solo hay {p.stock} unidades.")

    p.stock -= venta.quantity
    total    = round(p.precio * venta.quantity, 2)
    db.add(Venta(producto_id=venta.product_id, cantidad=venta.quantity, total=total))
    db.commit()
    nombre, stock_restante = p.nombre, p.stock
    db.close()
    return {
        "success":         True,
        "message":         f"Venta registrada: {venta.quantity} {nombre} por {MONEDA}{total:.2f}",
        "total":           total,
        "remaining_stock": stock_restante,
    }


@app.post("/inventario/entrada")
def registrar_entrada(entrada: EntradaCreate):
    db = SessionLocal()
    p  = db.query(Producto).filter(Producto.id == entrada.product_id).first()
    if not p:
        db.close()
        raise HTTPException(404, "Producto no encontrado")

    p.stock += entrada.quantity
    db.add(EntradaInventario(producto_id=entrada.product_id, cantidad=entrada.quantity))
    db.commit()
    nuevo_stock, nombre = p.stock, p.nombre
    db.close()
    return {
        "success":   True,
        "message":   f"Entrada: {entrada.quantity} {nombre}. Total en bodega: {nuevo_stock}",
        "new_stock": nuevo_stock,
    }


@app.post("/audio/{contexto}")
async def procesar_audio(contexto: str, audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    filename    = audio.filename or "recording.webm"

    try:
        transcripcion = transcribir_audio(audio_bytes, filename)
    except Exception as e:
        raise HTTPException(500, f"Error STT: {str(e)}")

    db       = SessionLocal()
    productos = db.query(Producto).all()
    db.close()

    try:
        datos = extraer_datos_audio(transcripcion, contexto, productos)
    except Exception as e:
        raise HTTPException(500, f"Error LLM: {str(e)}")

    print(f"datos procesador audio: {datos}")

    datos["transcription"] = transcripcion
    datos["success"]       = True

    # if contexto == "consulta":
    #     db       = SessionLocal()
    #     productos = db.query(Producto).all()
    #     db.close()
    #     sin_stock  = [p.nombre for p in productos if p.stock == 0]
    #     poco_stock = [p.nombre for p in productos if 0 < p.stock <= 5]
    #     partes = []
    #     if sin_stock:  partes.append(f"Sin stock: {', '.join(sin_stock)}")
    #     if poco_stock: partes.append(f"Poco stock: {', '.join(poco_stock)}")
    #     datos["message"]  = ". ".join(partes) if partes else "Todo el inventario esta bien."
    #     datos["products"] = [producto_to_dict(p) for p in productos]

    return datos


# ─── REPORTES ─────────────────────────────────────────────────

MESES_ES = ["","Enero","Febrero","Marzo","Abril","Mayo","Junio",
            "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]


@app.get("/reportes/mes")
def reporte_mes(mes: int = Query(..., ge=1, le=12), anio: int = Query(...)):
    db     = SessionLocal()
    ventas = db.query(Venta).filter(
        extract("month", Venta.fecha) == mes,
        extract("year",  Venta.fecha) == anio,
    ).all()

    total_monto  = round(sum(v.total for v in ventas), 2)
    top_producto = "-"
    if ventas:
        top_id  = Counter(v.producto_id for v in ventas).most_common(1)[0][0]
        prod    = db.query(Producto).filter(Producto.id == top_id).first()
        if prod:
            top_producto = prod.nombre
    db.close()
    return {
        "mes":          mes,
        "anio":         anio,
        "total_ventas": len(ventas),
        "total_monto":  total_monto,
        "top_producto": top_producto,
    }


@app.get("/reportes/pdf")
def reporte_pdf(mes: int = Query(..., ge=1, le=12), anio: int = Query(...)):
    db            = SessionLocal()
    ventas        = db.query(Venta).filter(
        extract("month", Venta.fecha) == mes,
        extract("year",  Venta.fecha) == anio,
    ).order_by(Venta.fecha).all()
    productos_map = {p.id: p for p in db.query(Producto).all()}
    db.close()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    estilos     = getSampleStyleSheet()
    naranja     = colors.HexColor("#FF6B2B")
    gris_texto  = colors.HexColor("#666666")
    negro       = colors.HexColor("#1A1A1A")
    fondo_claro = colors.HexColor("#FFF8F5")
    fondo_filas = colors.HexColor("#FFF0E8")

    s_titulo = ParagraphStyle("titulo", parent=estilos["Title"],
                               fontSize=24, spaceAfter=2, textColor=naranja)
    s_subtit = ParagraphStyle("subtit", parent=estilos["Normal"],
                               fontSize=12, spaceAfter=18, textColor=gris_texto)
    s_normal = ParagraphStyle("normal", parent=estilos["Normal"], fontSize=10)
    s_pie    = ParagraphStyle("pie",    parent=estilos["Normal"],
                               fontSize=9, textColor=colors.HexColor("#999999"))

    elementos = []

    # Encabezado
    elementos.append(Paragraph("Mi Tienda", s_titulo))
    elementos.append(Paragraph(
        f"Reporte de ventas &mdash; {MESES_ES[mes]} {anio}", s_subtit))
    elementos.append(HRFlowable(
        width="100%", thickness=2, color=naranja, spaceAfter=14))

    # Resumen
    total_monto   = round(sum(v.total for v in ventas), 2)
    total_ventas  = len(ventas)
    promedio      = round(total_monto / total_ventas, 2) if total_ventas else 0.0

    resumen_data = [
        ["Total de ventas",  str(total_ventas)],
        ["Monto total",      f"{MONEDA} {total_monto:.2f}"],
        ["Promedio por venta", f"{MONEDA} {promedio:.2f}"],
    ]
    resumen_tbl = Table(resumen_data, colWidths=[10*cm, 6*cm])
    resumen_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), fondo_claro),
        ("ROWBACKGROUNDS",(0,0), (-1,-1),  [fondo_claro, fondo_filas]),
        ("TEXTCOLOR",    (0, 0), (0, -1),  gris_texto),
        ("TEXTCOLOR",    (1, 0), (1, -1),  negro),
        ("FONTNAME",     (1, 0), (1, -1),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 11),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    elementos.append(resumen_tbl)
    elementos.append(Spacer(1, 20))

    # Tabla de ventas
    if ventas:
        cabecera = [
            Paragraph("<b>#</b>",       s_normal),
            Paragraph("<b>Fecha</b>",   s_normal),
            Paragraph("<b>Producto</b>",s_normal),
            Paragraph("<b>Cant.</b>",   s_normal),
            Paragraph("<b>Total</b>",   s_normal),
        ]
        filas = [cabecera]
        for i, v in enumerate(ventas, 1):
            prod   = productos_map.get(v.producto_id)
            nombre = prod.nombre if prod else "Desconocido"
            filas.append([
                Paragraph(str(i),                         s_normal),
                Paragraph(v.fecha.strftime("%d/%m %H:%M"),s_normal),
                Paragraph(nombre,                          s_normal),
                Paragraph(str(v.cantidad),                s_normal),
                Paragraph(f"{MONEDA} {v.total:.2f}",      s_normal),
            ])
        # Fila de total
        s_bold = ParagraphStyle("bold", parent=s_normal, fontName="Helvetica-Bold",
                                textColor=naranja)
        filas.append([
            Paragraph("", s_normal),
            Paragraph("", s_normal),
            Paragraph("", s_normal),
            Paragraph("TOTAL", s_bold),
            Paragraph(f"{MONEDA} {total_monto:.2f}", s_bold),
        ])

        col_widths = [1.2*cm, 3.5*cm, 7*cm, 2*cm, 3*cm]
        tbl = Table(filas, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            # Cabecera naranja
            ("BACKGROUND",    (0, 0), (-1, 0), naranja),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 10),
            # Filas alternas
            ("ROWBACKGROUNDS",(0, 1), (-1, -2), [colors.white, fondo_claro]),
            # Fila total
            ("BACKGROUND",    (0, -1), (-1, -1), fondo_filas),
            # Borde general
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#EEEEEE")),
            ("LINEBELOW",     (0, 0), (-1, 0),  1, naranja),
            # Padding
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            # Alinear columnas numéricas a la derecha
            ("ALIGN",         (3, 0), (4, -1),  "RIGHT"),
        ]))
        elementos.append(tbl)
    else:
        elementos.append(Paragraph(
            "No hay ventas registradas en este periodo.", s_normal))

    # Pie
    elementos.append(Spacer(1, 20))
    elementos.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#DDDDDD"), spaceAfter=8))
    elementos.append(Paragraph(
        f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')} &mdash; Mi Tienda",
        s_pie))

    doc.build(elementos)
    buf.seek(0)

    nombre_archivo = f"reporte-{anio}-{str(mes).zfill(2)}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{nombre_archivo}"'},
    )


@app.get("/reportes/ventas-hoy")
def ventas_hoy():
    db  = SessionLocal()
    hoy = datetime.now().date()
    vs  = db.query(Venta).filter(
        Venta.fecha >= datetime(hoy.year, hoy.month, hoy.day)
    ).all()
    total = round(sum(v.total for v in vs), 2)
    db.close()
    return {"total_ventas": len(vs), "total_monto": total, "fecha": str(hoy)}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook():
    return {"status": "webhook activo"}


# ─── SERVIR FRONTEND (va al final, después de todas las rutas API) ─────────
STATIC_DIR = Path(__file__).parent
if (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)