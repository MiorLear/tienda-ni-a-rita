# Mi Tienda — MVP PWA + Backend Python

Sistema de registro de ventas e inventario para personas analfabetas,
operado por voz e iconos grandes.

---

## Arquitectura

```
[PWA (index.html)]  ←→  [FastAPI (main.py)]  ←→  [SQLite]
                              ↕
                    [Whisper API (audio→texto)]
                    [GPT-4o-mini (texto→intención)]
                              ↕
                    [Bot WhatsApp (Twilio / Meta)]
```

---

## Setup rápido

### 1. Backend Python

```bash
pip install fastapi uvicorn python-multipart openai sqlalchemy python-dotenv

# Crear .env
echo "OPENAI_API_KEY=sk-..." > .env

uvicorn main:app --reload --port 8000
```

El backend ya incluye datos de demo y funciona sin API key (modo simulado).

### 2. PWA (Frontend)

Sirve los archivos estáticos con cualquier servidor:

```bash
# Con Python
python -m http.server 3000

# Con Node
npx serve . -p 3000
```

O despliégala en **Vercel / Netlify** apuntando a la carpeta.

### 3. Cambiar URL del backend

En `index.html`, línea ~215:
```js
const API_BASE = 'http://localhost:8000'; // ← Cambia a tu URL
```

---

## Endpoints del API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/inventario` | Lista todos los productos con stock |
| POST | `/ventas` | Registra una venta `{product_id, quantity}` |
| POST | `/inventario/entrada` | Agrega stock `{product_id, quantity}` |
| POST | `/audio/{contexto}` | Procesa audio (venta/entrada/consulta) |
| GET | `/reportes/ventas-hoy` | Resumen del día |
| POST | `/webhook/whatsapp` | Webhook del bot de WhatsApp |

---

## Flujo de audio

```
Usuario graba → /audio/venta
                    ↓
              Whisper transcribe
                    ↓
              GPT-4o-mini extrae:
              - producto detectado
              - cantidad
              - intención
                    ↓
              Frontend pre-llena los campos
                    ↓
              Usuario confirma con botón grande
```

---

## Bot de WhatsApp

Para conectar WhatsApp tienes dos opciones:

### Opción A — Twilio Sandbox (rápido para MVP)
```bash
pip install twilio
# Configura ngrok para exponer localhost:8000
ngrok http 8000
# En Twilio Sandbox, configura el webhook a:
# https://xxxx.ngrok.io/webhook/whatsapp
```

### Opción B — Meta Cloud API (oficial, gratis)
- Crear app en developers.facebook.com
- Configurar Webhooks → verify_token
- Número de WhatsApp Business

---

## Agregar productos

En `main.py`, función `seed_db()`, agrega productos al array:
```python
Producto(nombre="Leche", icono="🥛", stock=20, precio=10.0),
```

---

## Próximos pasos para el MVP

- [ ] Sincronización offline (cola en IndexedDB)
- [ ] Autenticación simple por PIN (4 dígitos con iconos)
- [ ] Reportes por voz ("¿Cuánto vendí hoy?")
- [ ] Fotos de productos como iconos
- [ ] Alertas de stock bajo por WhatsApp