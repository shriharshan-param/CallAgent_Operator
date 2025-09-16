from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse
import uvicorn
from aistudio_operator import voice_gateway

app = FastAPI()

@app.post("/twilio/voice")
async def handle_twilio_voice(request: Request):
    """Handle Twilio voice webhook"""
    try:
        response = await voice_gateway.twilio_voice_webhook(request)
        return PlainTextResponse(content=response, media_type="application/xml")
    except Exception as e:
        return PlainTextResponse(content=str(e), media_type="application/xml")

@app.websocket("/hume/ws/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    """WebSocket endpoint for Hume AI integration"""
    try:
        await voice_gateway.hume_websocket_endpoint(websocket, call_sid)
    except Exception:
        await websocket.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")