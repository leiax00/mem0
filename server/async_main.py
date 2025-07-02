import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Security
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.requests import Request

from mem0 import AsyncMemory

DEBUG = os.getenv("DEBUG", "") in ["1", "true"]
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY", "")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": "gpt-4o"}},
    "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}},
    "history_db_path": HISTORY_DB_PATH,
}

config_path = os.environ.get("CONFIG_PATH", "./config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as rf:
        DEFAULT_CONFIG = yaml.safe_load(rf)

custom_fact_extraction_prompt = DEFAULT_CONFIG.get("custom_fact_extraction_prompt")
if custom_fact_extraction_prompt is not None:
    DEFAULT_CONFIG["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt.replace(
        "${ENV_CUR_TIME}",
        datetime.now().strftime("%Y-%m-%d")
    )
logging.info(f"config: {json.dumps(DEFAULT_CONFIG, indent=4, ensure_ascii=False)}")


def get_memory_instance(request: Request) -> AsyncMemory:
    return request.app.state.memory_instance


memory_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(main: FastAPI):
    try:
        async with memory_lock:
            main.state.memory_instance = await AsyncMemory.from_config(DEFAULT_CONFIG)
        yield
    except Exception as e:
        logging.exception(f"Exception occurred: {e}")


app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
    lifespan=lifespan,
)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def verify_api_key(
        authorization: Optional[str] = Security(api_key_header),
):
    if API_KEY != "":
        if authorization is None or not authorization.startswith("Token "):
            raise HTTPException(status_code=401, detail="Unauthorized")

        token = authorization.replace("Token ", "").strip()
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="Unauthorized")


v1_router = APIRouter(tags=["v1"], dependencies=[Depends(verify_api_key)])
v2_router = APIRouter(tags=["v2"], dependencies=[Depends(verify_api_key)])


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


@v1_router.get("/ping/", summary="ping server")
async def ping():
    return {
        "org_id": 0,
        "project_id": 0,
        "user_email": "wewins@we-wins.com"
    }


@v1_router.post("/configure", summary="Configure Mem0")
async def set_config(config: Dict[str, Any], request: Request):
    """Set memory configuration."""
    async with memory_lock:
        request.app.state.memory_instance = await AsyncMemory.from_config(config)
    return {"message": "Configuration set successfully"}


@v1_router.post("/memories/", summary="Create memories")
async def add_memory(memory_create: MemoryCreate, memory: AsyncMemory = Depends(get_memory_instance)):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = await memory.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/memories/", summary="Get memories")
async def get_all_memories(
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory: AsyncMemory = Depends(get_memory_instance),
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return await memory.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/memories/{memory_id}/", summary="Get a memory")
async def get_memory(memory_id: str, memory: AsyncMemory = Depends(get_memory_instance)):
    """Retrieve a specific memory by ID."""
    try:
        return await memory.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.post("/memories/search/", summary="Search memories")
async def search_memories(search_req: SearchRequest, memory: AsyncMemory = Depends(get_memory_instance)):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return await memory.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.put("/memories/{memory_id}/", summary="Update a memory")
async def update_memory(memory_id: str, updated_memory: Dict[str, Any], memory: AsyncMemory = Depends(get_memory_instance)):
    """Update an existing memory."""
    try:
        return await memory.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/memories/{memory_id}/history/", summary="Get memory history")
async def memory_history(memory_id: str, memory: AsyncMemory = Depends(get_memory_instance)):
    """Retrieve memory history."""
    try:
        return await memory.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.delete("/memories/{memory_id}/", summary="Delete a memory")
async def delete_memory(memory_id: str, memory: AsyncMemory = Depends(get_memory_instance)):
    """Delete a specific memory by ID."""
    try:
        await memory.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.delete("/memories/", summary="Delete all memories")
async def delete_all_memories(
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory: AsyncMemory = Depends(get_memory_instance),
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        await memory.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.post("/reset/", summary="Reset all memories")
async def reset_memory(memory: AsyncMemory = Depends(get_memory_instance)):
    """Completely reset stored memories."""
    try:
        await memory.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.get("/entities/", summary="获取所有类型的用户")
async def get_users():
    # todo 暂未实现
    return []


@v2_router.delete("/entities/{mem_type}/{name}/", summary="删除指定类型的记忆")
async def delete_users(mem_type: str, name: str, memory: AsyncMemory = Depends(get_memory_instance)):
    """Delete all users for a given memory type."""
    params = {f"{mem_type}_id": name}
    try:
        return await memory.delete_all(**params)
    except Exception as e:
        logging.exception("Error in delete memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
async def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")


app.include_router(v1_router, prefix="/v1")
app.include_router(v2_router, prefix="/v2")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
