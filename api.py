# api.py

from fastapi import FastAPI
from tools import tool_brainstorm_claims  # Import your tool router here

app = FastAPI()

# Include tool routers here
app.include_router(tool_brainstorm_claims.router)