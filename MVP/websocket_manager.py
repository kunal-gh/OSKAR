import json
from typing import List, Dict, Any
from fastapi import WebSocket

class WebSocketManager:
    """
    v0.8 WebSocket Connection Manager.
    Maintains a register of active dashboard clients.
    Broadcasts parsed OSKAR risk evaluation results in real-time.
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] Client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket] Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcasts a JSON dictionary to all connected clients.
        """
        payload = json.dumps(message)
        
        # Create a copy to iterate safely in case a client disconnects during broadcast
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception as e:
                print(f"[WebSocket] Error broadcasting to client: {e}")
                self.disconnect(connection)

# Global singleton instance
ws_manager = WebSocketManager()
