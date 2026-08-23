"""
Voice Bridge Server
-------------------
Runs on Hetzner. Receives audio from the phone, transcribes with Whisper,
calls the LLM (FreeLLMAPI), executes tools if requested, and streams
logs/replies. Serves a PWA client.

Kernel tools (kernel_status/kernel_decide/kernel_memo) talk to the thrivbe-os
kernel voice HTTP surface running on Robin's Mac over Tailscale.

Usage:
  /opt/voice-bridge/bin/uvicorn server:app --host 0.0.0.0 --port 8765 --ssl-keyfile certs/cert.key --ssl-certfile certs/cert.crt
"""

import os
import json
import tempfile
import subprocess
import asyncio
import httpx
import time
import base64
import shlex
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse, FileResponse
from faster_whisper import WhisperModel

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Voice Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Environment Variables ──────────────────────────────────────────────
def load_env():
    env_path = "/opt/voice-bridge/.env"
    if os.path.exists(env_path):
        print(f"Loading env from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip()

load_env()

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8792/v1/chat/completions"
API_KEY = os.getenv("FREELLMAPI_KEY")
DEFAULT_MODEL = "auto"
VOICE_BRIDGE_TOKEN = os.getenv("VOICE_BRIDGE_TOKEN")

# thrivbe-os Kernel Voice API Configuration (runs on Robin's Mac, over Tailscale)
KERNEL_BASE_URL = "http://127.0.0.1:8790"

# ElevenLabs Premium TTS Configuration
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel (Premium, highly realistic)
ELEVENLABS_MODEL = "eleven_turbo_v2_5"        # Low latency, high quality

SSH_CMD = ["ssh", "-p", "8022", "-o", "StrictHostKeyChecking=no", "-i", "/root/.ssh/id_ed25519", "u0_a634@100.127.4.13"]

def run_ssh(remote_args, capture=False):
    remote_cmd = " ".join(shlex.quote(a) for a in remote_args)
    if capture:
        return subprocess.check_output(SSH_CMD + [remote_cmd], text=True)
    subprocess.check_call(SSH_CMD + [remote_cmd])

def require_voice_token(x_voice_token: str = Header(default=None)):
    if not VOICE_BRIDGE_TOKEN or x_voice_token != VOICE_BRIDGE_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

async def synthesize_speech(text: str) -> str:
    """Synthesizes text to speech using ElevenLabs and returns base64 MP3 data."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("[TTS] ELEVENLABS_API_KEY not configured. Skipping premium voice.")
        return ""
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": api_key,
        "content-type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    try:
        print(f"[TTS] Requesting ElevenLabs synthesis for: '{text[:40]}...'")
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code == 200:
                audio_bytes = r.content
                print(f"[TTS] ElevenLabs audio generated successfully ({len(audio_bytes)} bytes).")
                return base64.b64encode(audio_bytes).decode("utf-8")
            else:
                print(f"[TTS ERROR] ElevenLabs returned {r.status_code}: {r.text}")
                return ""
    except Exception as e:
        print(f"[TTS EXCEPTION] ElevenLabs failed: {e}")
        return ""

SYSTEM_PROMPT = (
    "You are an agentic voice assistant for Robin. "
    "You have tools to monitor and control his Android phone (initiating calls, reading notifications, setting the clipboard, searching contacts), "
    "interact with Bloom (his agent-native task platform), search Twenty CRM contacts, "
    "perform semantic search queries (using turbovec) across his LinkedIn network ('people' corpus), Notion knowledge base ('notion' corpus), and Obsidian wiki/agent skills ('wiki_skills' corpus), "
    "read his Inbox items and inspect Command Center graph documents. "
    "You can also perform a Hybrid RAG search (using 'hybrid_rag_search') combining system graph relations and vector search over all wiki pages, skills, and codebases to find detailed conceptual answers, matching entities, and verified graph paths. "
    "Note: All book summaries, book abstracts, and library documents (like getAbstract summaries) are stored in the Obsidian library under the 'wiki_skills' corpus. Always use 'wiki_skills' (or 'hybrid_rag_search') to search for books, book summaries, or abstracts; do not search the 'notion' corpus for these. If you need to search for tasks, projects, or objectives in the Command Center (e.g. for keywords like 'smart rings' or general project queries), use 'hybrid_rag_search' to query the system graph and database. "
    "You also have six kernel tools that talk directly to the thrivbe-os kernel running on Robin's Mac: 'kernel_status' tells you what needs Robin's attention right now (pending approvals, attention items, and recent agent runs); 'kernel_decide' approves or rejects a specific pending approval by its id; 'kernel_memo' files a short note into Robin's unified inbox; 'kernel_remember' stores a durable fact in Robin's ONE memory; 'kernel_recall' searches that memory; 'os_delegate' hands business/system work to the OS worker and returns immediately after acceptance. "
    "CRITICAL: Never call 'kernel_decide' based on inference. Only call it after you have read the approval's preview text aloud to Robin and he has given an explicit yes or no. If 'kernel_decide' reports the approval was already decided or doesn't exist, tell Robin that plainly. "
    "Keep responses short and conversational. No markdown, no bullet points, no code blocks. "
    "CRITICAL: If you are returning lists of names, search results, contacts, or notification details: "
    "speak only a very brief summary or count before a '===' divider (e.g. 'I found 3 contacts matching Robin. Here is the list. ==='), "
    "and list the full details AFTER the '===' divider. The speech engine will only read out the short text before the divider, "
    "while the complete detailed list will be displayed in the Web App interface logs."
)

LOCAL_SYSTEM_PROMPT_ADDENDUM = (
    "You have tools to monitor and control his Android phone (initiating calls, reading notifications, setting the clipboard, searching contacts), "
    "interact with Bloom (his agent-native task platform), search Twenty CRM contacts, "
    "perform semantic search queries (using turbovec) across his LinkedIn network ('people' corpus), Notion knowledge base ('notion' corpus), and Obsidian wiki/agent skills ('wiki_skills' corpus), "
    "read his Inbox items and inspect Command Center graph documents. "
    "You can also perform a Hybrid RAG search (using 'hybrid_rag_search') combining system graph relations and vector search over all wiki pages, skills, and codebases to find detailed conceptual answers, matching entities, and verified graph paths. "
    "Note: All book summaries, book abstracts, and library documents (like getAbstract summaries) are stored in the Obsidian library under the 'wiki_skills' corpus. Always use 'wiki_skills' (or 'hybrid_rag_search') to search for books, book summaries, or abstracts; do not search the 'notion' corpus for these. If you need to search for tasks, projects, or objectives in the Command Center (e.g. for keywords like 'smart rings' or general project queries), use 'hybrid_rag_search' to query the system graph and database. "
    "You also have six kernel tools that talk directly to the thrivbe-os kernel running on Robin's Mac: 'kernel_status' tells you what needs Robin's attention right now (pending approvals, attention items, and recent agent runs); 'kernel_decide' approves or rejects a specific pending approval by its id; 'kernel_memo' files a short note into Robin's unified inbox; 'kernel_remember' stores a durable fact in Robin's ONE memory; 'kernel_recall' searches that memory; 'os_delegate' hands business/system work to the OS worker and returns immediately after acceptance. "
    "CRITICAL: Never call 'kernel_decide' based on inference. Only call it after you have read the approval's preview text aloud to Robin and he has given an explicit yes or no. If 'kernel_decide' reports the approval was already decided or doesn't exist, tell Robin that plainly. "
    "Keep responses short and conversational. No markdown, no bullet points, no code blocks. "
    "CRITICAL: If you are returning lists of names, search results, contacts, or notification details: "
    "speak only a very brief summary or count before a '===' divider (e.g. 'I found 3 contacts matching Robin. Here is the list. ==='), "
    "and list the full details AFTER the '===' divider. The speech engine will only read out the short text before the divider, "
    "while the complete detailed list will be displayed in the Web App interface logs."
)

SYSTEM_PROMPT_SEPARATOR = "\n\n--- LOCAL VOICE BRIDGE TOOL GUIDANCE ---\n\n"
SYSTEM_PROMPT_TTL_SECONDS = 600
_system_prompt_cache_text = None
_system_prompt_cache_fetched_at = 0

async def get_system_prompt() -> str:
    global _system_prompt_cache_text, _system_prompt_cache_fetched_at

    now = time.time()
    if (
        _system_prompt_cache_text
        and now - _system_prompt_cache_fetched_at < SYSTEM_PROMPT_TTL_SECONDS
    ):
        return _system_prompt_cache_text + SYSTEM_PROMPT_SEPARATOR + LOCAL_SYSTEM_PROMPT_ADDENDUM

    token = os.getenv("VOICE_API_TOKEN")
    if token:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{KERNEL_BASE_URL}/persona",
                    headers={"Authorization": f"Bearer {token}"}
                )
                r.raise_for_status()
                data = r.json()
                persona = data.get("persona")
                if isinstance(persona, str) and persona.strip():
                    _system_prompt_cache_text = persona.strip()
                    _system_prompt_cache_fetched_at = now
        except Exception as e:
            print(f"[SYSTEM PROMPT] Failed to fetch kernel persona: {e}")

    if _system_prompt_cache_text:
        return _system_prompt_cache_text + SYSTEM_PROMPT_SEPARATOR + LOCAL_SYSTEM_PROMPT_ADDENDUM
    return SYSTEM_PROMPT

# ── Whisper Setup ───────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print(f"Whisper ({WHISPER_MODEL_SIZE}) ready.")

# ── Dynamic Self-Learning Registry ──────────────────────────────────────────
REGISTRY_PATH = "/opt/voice-bridge/dynamic_registry.json"

def load_dynamic_tools():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dynamic registry: {e}")
        return {}

DYNAMIC_TOOLS = load_dynamic_tools()

# ── Base Tools list ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_phone_status",
            "description": "Get current battery level, charging status, and temperature of Robin's Android phone.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vibrate_phone",
            "description": "Vibrate Robin's Android phone to get attention.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration in milliseconds to vibrate, default is 500."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send an SMS text message to a contact from Robin's phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_number": {
                        "type": "string",
                        "description": "The recipient's phone number."
                    },
                    "message": {
                        "type": "string",
                        "description": "The text message content."
                    }
                },
                "required": ["to_number", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kernel_status",
            "description": "Check what needs Robin's attention right now: pending approvals awaiting his decision, attention items surfaced by the OS, and recent agent runs. Use this to answer questions like 'what needs my attention', 'what's pending', or 'what has the OS been doing'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kernel_decide",
            "description": "Approve or reject a pending kernel approval by id. Only call this after reading the approval's preview text aloud to Robin and receiving an explicit yes or no from him — never decide on inference alone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "approval_id": {
                        "type": "integer",
                        "description": "The approval id, from a prior kernel_status call."
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["approved", "rejected"],
                        "description": "Robin's explicit decision."
                    }
                },
                "required": ["approval_id", "decision"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kernel_memo",
            "description": "File a short note into Robin's unified inbox. Use for requests like 'note that down', 'remind me', or 'add this to my inbox'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memo text, 1-4000 characters."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kernel_remember",
            "description": "Store a durable fact in Robin's ONE memory (shared across voice, Telegram, and the OS). Use for 'remember that...', preferences, decisions, or context worth keeping. Distinct from kernel_memo, which files an inbox item for triage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The fact to remember, 1-4000 characters."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kernel_recall",
            "description": "Search Robin's ONE memory (remembered notes first, then CRM/tasks/wiki) for facts or context. Use for 'what did I say about...', 'do you remember...', or any question about stored knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "os_delegate",
            "description": "Hand BUSINESS/SYSTEM work to Robin's thrivbe-os worker: tasks, CRM updates, approvals, follow-ups/chasing, or anything in Robin's operating system such as 'create a task', 'update...', or 'chase...'. This only waits for the OS to accept the job; Robin gets a Telegram approval or summary later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "The business/system instruction for the OS worker."
                    }
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bloom_list_projects",
            "description": "List all active/visible projects in Bloom (Asana Clone) to find project IDs.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bloom_list_tasks",
            "description": "List tasks in the Bloom/thrivbe-work project board. Note: If you do not find the task here, or if you want to search globally across all databases, workspaces, and wiki pages, you must fall back to 'hybrid_rag_search'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "projectId": {
                        "type": "integer",
                        "description": "The project ID to filter tasks by."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["todo", "in_progress", "done"],
                        "description": "The task status to filter by."
                    },
                    "q": {
                        "type": "string",
                        "description": "Search query text to filter tasks."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bloom_create_task",
            "description": "Request creation of a new task in a Bloom project. This is gated: it files an approval for Robin and does not take effect until he approves it in Telegram — tell him that plainly rather than saying the task is created.",
            "parameters": {
                "type": "object",
                "properties": {
                    "projectId": {
                        "type": "integer",
                        "description": "The ID of the project to add the task to."
                    },
                    "title": {
                        "type": "string",
                        "description": "The title of the task."
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task."
                    }
                },
                "required": ["projectId", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bloom_update_task",
            "description": "Request an update to an existing task in Bloom (e.g. status, title). This is gated: it files an approval for Robin and does not take effect until he approves it in Telegram — tell him that plainly rather than saying the task is updated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The task ID to update."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["todo", "in_progress", "done"],
                        "description": "Set task status."
                    },
                    "title": {
                        "type": "string",
                        "description": "New title for the task."
                    }
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bloom_comment_task",
            "description": "Request adding a comment to an existing task in Bloom. This is gated: it files an approval for Robin and does not take effect until he approves it in Telegram — tell him that plainly rather than saying the comment is posted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The task ID to comment on."
                    },
                    "body": {
                        "type": "string",
                        "description": "The comment text."
                    }
                },
                "required": ["id", "body"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "graph_get_node",
            "description": "Inspect a specific node's full context (neighborhood) in the system graph to see related files, agents, workflows, or connections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The node ID (e.g. 'wiki_page:undavos')."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Relevance depth (0-3), default is 1."
                    }
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graph_get_document",
            "description": "Read the full contents of a markdown document or wiki page from the Command Center catalog using its node ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The node ID (e.g. 'wiki_page:undavos')."
                    }
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_inbox_items",
            "description": "List pending or triaged inbox items (like emails, SMS, Beeper notifications, LinkedIn messages) from the Command Center.",
            "parameters": {
                "type": "object",
                "properties": {
                    "triage_status": {
                        "type": "string",
                        "enum": ["unprocessed", "triaged", "snoozed", "archived", "actioned", "all"],
                        "description": "Filter by triage status, default is 'unprocessed'."
                    },
                    "search": {
                        "type": "string",
                        "description": "Optional search query to filter items by sender name, subject, or snippet content."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of items to return (default 15)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semsearch_query",
            "description": "Run semantic search queries against Robin's LinkedIn connections ('people' corpus), Notion knowledge base ('notion' corpus), or Obsidian wiki/agent skills ('wiki_skills' corpus).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language query (e.g., 'climate tech founders in Norway' or 'insights about vector quantization')."
                    },
                    "corpus": {
                        "type": "string",
                        "enum": ["people", "notion", "wiki_skills"],
                        "description": "The target corpus. 'people' searches LinkedIn connections, 'notion' searches the Notion AAA knowledge base, 'wiki_skills' searches the Obsidian wiki and agent skills. Defaults to 'people'."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_rag_search",
            "description": "Perform a Hybrid RAG search combining system graph relations and vector search over all wiki pages, skills, and codebases. Returns a direct conceptual answer, matching entities, and verified graph paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language query or question (e.g. 'what is the sync pipeline configuration' or 'who depends on hybrid rag')."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "twenty_search_contacts",
            "description": "Search for contacts (people) by first or last name in Twenty CRM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_query": {
                        "type": "string",
                        "description": "The name or part of the name to search for (e.g. 'Robin' or 'Sverd')."
                    }
                },
                "required": ["name_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "android_list_contacts",
            "description": "Search for contact names and phone numbers inside Robin's Android phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_query": {
                        "type": "string",
                        "description": "Optional contact name to filter by."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "android_make_call",
            "description": "Initiate an active phone call to a specific phone number from Robin's Android phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "The telephone number to call."
                    }
                },
                "required": ["phone_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "android_set_clipboard",
            "description": "Copy a text block to the clipboard of Robin's Android phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to copy."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "android_get_notifications",
            "description": "Read the latest active notifications from Robin's Android phone.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ── JSON Tool Arguments Helper ──────────────────────────────────────────────
def clean_and_parse_json(s: str, tool_name: str = None) -> dict:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        
    try:
        val = json.loads(s)
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            return wrap_primitive_arguments(val, tool_name)
    except Exception:
        pass

    if not (s.startswith("{") and s.endswith("}")):
        cleaned = s.strip('"\'')
        return wrap_primitive_arguments(cleaned, tool_name)

    import ast
    try:
        val = ast.literal_eval(s)
        if isinstance(val, dict):
            return {str(k): v for k, v in val.items()}
    except Exception:
        pass

    import re
    for key in ["name_query", "query", "command", "phone_number", "id", "text", "to_number", "message", "projectId", "title"]:
        match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', s)
        if match:
            return {key: match.group(1)}
        match = re.search(rf'[\'"]?{key}[\'"]?\s*:\s*[\'"]([^\'"]+)[\'"]', s)
        if match:
            return {key: match.group(1)}

    raise ValueError(f"Could not parse tool arguments as JSON: {s}")

def wrap_primitive_arguments(val: str, tool_name: str) -> dict:
    mapping = {
        "twenty_search_contacts": "name_query",
        "android_list_contacts": "name_query",
        "hybrid_rag_search": "query",
        "semsearch_query": "query",
        "android_make_call": "phone_number",
        "android_set_clipboard": "text",
        "vibrate_phone": "duration_ms",
        "bloom_list_tasks": "q",
    }
    key = mapping.get(tool_name, "query")
    if key == "duration_ms":
        try:
            return {key: int(val)}
        except ValueError:
            return {key: 500}
    return {key: val}


# ── Tool Execution logic ────────────────────────────────────────────────────
def normalize_tool_args(args: dict) -> dict:
    """LLMs emit variant/malformed keys (':title', 'project_id'). Normalize before
    tool code does strict key access — seen crashing bloom_create_task 2026-07-10."""
    aliases = {"project_id": "projectId", "projectid": "projectId", "task_id": "id", "taskid": "id"}
    out = {}
    for k, v in args.items():
        if isinstance(k, str):
            k = k.strip().strip(":'\" ")
            k = aliases.get(k.lower(), k)
        out[k] = v
    return out


async def _bloom_project_ids(token: str) -> dict:
    """id -> name of Bloom projects, via the kernel."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{KERNEL_BASE_URL}/bloom-projects",
            headers={"Authorization": f"Bearer {token}"}
        )
        r.raise_for_status()
        data = r.json()
    projects = data.get("projects", data) if isinstance(data, dict) else data
    return {int(p["id"]): p["name"] for p in projects}


async def execute_tool(name: str, args: dict) -> str:
    args = normalize_tool_args(args)
    print(f"[TOOL] Executing {name} with args {args}")
    try:
        # 1. Check Dynamic Tools first
        if name in DYNAMIC_TOOLS:
            if name in globals():
                func = globals()[name]
                if asyncio.iscoroutinefunction(func):
                    res = await func(**args)
                else:
                    res = func(**args)
                return str(res)
            else:
                return f"Error: Tool {name} is registered but its execution namespace is missing."

        # 2. Check Static Tools
        if name == "get_phone_status":
            res = run_ssh(["termux-battery-status"], capture=True)
            data = json.loads(res)
            return f"Battery is {data['percentage']}% ({data['status']}), temperature is {data['temperature']}°C."
            
        elif name == "vibrate_phone":
            # Bypassed SSH and handled natively by browser client
            return "Phone vibrated via browser haptics."
            
        elif name == "send_sms":
            to = args["to_number"]
            msg = args["message"]
            run_ssh(["termux-sms-send", "-n", to, msg], capture=False)
            return f"SMS successfully sent to {to}."
            
        elif name == "kernel_status":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/status",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            approvals = data.get("approvals", [])
            attention = data.get("attention", [])
            runs = data.get("runs", [])

            lines = []
            if approvals:
                lines.append("Pending approvals:")
                for a in approvals:
                    lines.append(f"- [{a['id']}] {a.get('actionType', 'action')}: {a.get('preview', '')}")
            if attention:
                lines.append("Needs attention:")
                for item in attention:
                    lines.append(f"- ({item.get('priority', '')}) {item.get('title', '')}: {item.get('detail', '')}")
            if runs:
                lines.append("Recent agent runs:")
                for run in runs:
                    lines.append(f"- [{run['id']}] {run.get('process_name', '')} — {run.get('status', '')}")

            return "\n".join(lines) if lines else "Nothing pending — no approvals, attention items, or runs."

        elif name == "kernel_decide":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            payload = {"approvalId": args["approval_id"], "decision": args["decision"]}
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(
                        f"{KERNEL_BASE_URL}/decide",
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            if data.get("changed"):
                return f"Done — approval {args['approval_id']} marked as {args['decision']}."
            return "That approval was already decided, or it doesn't exist."

        elif name == "kernel_memo":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(
                        f"{KERNEL_BASE_URL}/memo",
                        json={"text": args["text"]},
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            return f"Noted — filed to your inbox (id {data.get('id')})."

        elif name == "kernel_remember":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            text = (args.get("text") or "").strip()
            if not text:
                return "I need something to remember."
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(
                        f"{KERNEL_BASE_URL}/remember",
                        json={"text": text},
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"
            return "Remembered."

        elif name == "kernel_recall":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            q = (args.get("query") or "").strip()
            if not q:
                return "What should I recall?"
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/recall",
                        params={"q": q},
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"
            hits = data.get("hits", [])
            if not hits:
                return f"Nothing recalled for {q}."
            return "\n".join((h.get("text") or "")[:200] for h in hits[:4])

        elif name == "os_delegate":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            instruction = (args.get("instruction") or "").strip()
            if not instruction:
                return "I need an instruction to hand to the OS worker."
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(
                        f"{KERNEL_BASE_URL}/delegate",
                        json={"instruction": instruction, "source": "voice-bridge"},
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            return "Handed to the OS worker — Robin will get an approval or a summary on Telegram."

        # 3. Bloom Tools
        elif name == "bloom_list_projects":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/bloom-projects",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            projects = data.get("projects", data) if isinstance(data, dict) else data
            lines = []
            for p in projects:
                lines.append(f"- ID {p['id']}: {p['name']} ({p.get('organization_name', 'No Org')})")
            return "\n".join(lines) if lines else "No projects found in Bloom."

        elif name == "bloom_list_tasks":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            params = {}
            if "projectId" in args: params["projectId"] = args["projectId"]
            if "status" in args: params["status"] = args["status"]
            if "q" in args: params["q"] = args["q"]
            
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/bloom-tasks",
                        params=params,
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            tasks = data.get("tasks", [])
            lines = []
            for t in tasks:
                lines.append(f"- ID {t['id']}: {t['title']} [Status: {t['status']}] (Project ID: {t.get('project_id', 'None')})")
            return "\n".join(lines) if lines else "No matching tasks found."

        elif name in ("bloom_create_task", "bloom_update_task", "bloom_comment_task"):
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            if name == "bloom_create_task":
                missing = [k for k in ("projectId", "title") if k not in args]
                if missing:
                    return f"Error: bloom_create_task is missing required argument(s): {', '.join(missing)}. Provide them and retry."
                try:
                    known = await _bloom_project_ids(token)
                except Exception as e:
                    return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"
                try:
                    pid = int(args["projectId"])
                except (TypeError, ValueError):
                    pid = -1
                if pid not in known:
                    listing = "\n".join(f"- ID {i}: {n}" for i, n in known.items())
                    return f"Error: projectId {args['projectId']} doesn't exist in Bloom. Pick one of these real projects and retry:\n{listing}"
                payload = {"op": "create", "projectId": pid, "title": args["title"]}
                if "description" in args: payload["description"] = args["description"]
            elif name == "bloom_update_task":
                if "id" not in args:
                    return "Error: bloom_update_task is missing required argument 'id' (the task ID)."
                payload = {"op": "update", "id": args["id"]}
                if "status" in args: payload["status"] = args["status"]
                if "title" in args: payload["title"] = args["title"]
            else:
                missing = [k for k in ("id", "body") if k not in args]
                if missing:
                    return f"Error: bloom_comment_task is missing required argument(s): {', '.join(missing)}."
                payload = {"op": "comment", "id": args["id"], "body": args["body"]}
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(
                        f"{KERNEL_BASE_URL}/bloom-write",
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"
            return data.get("message", f"Filed approval #{data.get('approvalId')} — Robin must approve before this executes.")

        # 4. System Graph Tools


        elif name == "graph_get_node":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            node_id = args["id"]
            depth = args.get("depth", 1)
            params = {"id": node_id, "depth": depth, "maxNodes": 50}
            
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/graph-node",
                        params=params,
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"
            
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            
            node_map = {n["id"]: n for n in nodes}
            lines = []
            
            for e in edges:
                src_name = node_map.get(e["source"], {}).get("name", e["source"])
                tgt_name = node_map.get(e["target"], {}).get("name", e["target"])
                lines.append(f"- {src_name} --[{e['kind']}]--> {tgt_name}")
            
            target_node = node_map.get(node_id, {})
            target_desc = target_node.get("description", "No description available.")
            
            res_str = f"Node: {target_node.get('name', node_id)} ({node_id})\nDescription: {target_desc}\n\n"
            if lines:
                res_str += "Graph Connections:\n" + "\n".join(lines)
            else:
                res_str += "No connections found."
            return res_str

        elif name == "graph_get_document":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            doc_id = args["id"]
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/graph-doc",
                        params={"id": doc_id},
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            try:
                data = r.json()
                return data.get("content", "No content found.")
            except Exception:
                return r.text[:2000]

        elif name == "list_inbox_items":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            params = {"triage_status": args.get("triage_status", "unprocessed")}
            if "search" in args:
                params["search"] = args["search"]
            if "limit" in args:
                params["limit"] = args["limit"]
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/inbox",
                        params=params,
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            items = data.get("items", [])
            lines = []
            display_limit = args.get("limit", 15)
            for item in items[:display_limit]:
                sender = item.get("sender") or "System"
                subject = item.get("subject") or "No Subject"
                snippet = item.get("snippet") or ""
                lines.append(f"- [{item['id']}] From {sender}: {subject} ({snippet[:60]}...)")
            return "\n".join(lines) if lines else f"No {params['triage_status']} inbox items found."

        elif name == "semsearch_query":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            q = args["query"]
            corpus = args.get("corpus", "people")
            payload = {"query": q, "k": 5}
            if corpus != "people":
                payload["corpus"] = corpus

            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(
                        f"{KERNEL_BASE_URL}/semsearch",
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            results = data.get("results", [])
            lines = []
            for res in results:
                name_val = res.get("name") or res.get("title") or "Match"
                headline = res.get("headline") or res.get("snippet") or ""
                if not headline and "path" in res:
                    headline = f"Path: {res['path']} (chunk {res.get('chunk_index', 0)})"
                company = f" @ {res['company']}" if res.get("company") else ""
                lines.append(f"- {name_val}{company}: {headline[:80]}...")
            return "\n".join(lines) if lines else f"No semantic search results in {corpus}."

        elif name == "hybrid_rag_search":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            query = args["query"]
            payload = {"query": query}
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        f"{KERNEL_BASE_URL}/ai-search",
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=30.0
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            answer = data.get("answer", "")
            
            entities = data.get("entities", [])
            paths = data.get("graphPaths", [])
            
            res_lines = [f"Answer: {answer}"]
            if entities:
                res_lines.append("\nRelevant Entities:")
                for ent in entities:
                    res_lines.append(f"- {ent['title']} ({ent['type']}): {ent['relevance']}")
            if paths:
                res_lines.append("\nVerified Paths:")
                for p in paths:
                    path_str = " -> ".join(p.get("via", []))
                    res_lines.append(f"- {p['from']} to {p['to']} via: {path_str}")
            return "\n".join(res_lines)

        elif name == "twenty_search_contacts":
            token = os.getenv("VOICE_API_TOKEN")
            if not token:
                return "Error: VOICE_API_TOKEN is not configured in environment variables."
            q_val = args["name_query"]
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        f"{KERNEL_BASE_URL}/twenty-search",
                        params={"q": q_val},
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    r.raise_for_status()
                    data = r.json()
            except Exception as e:
                return f"I couldn't reach the Thrivbe OS kernel just now. ({e})"

            lines = []
            for p in data.get("results", []):
                email = p.get("email") or "No Email"
                job = p.get("jobTitle") or "Contact"
                lines.append(f"- {p.get('firstName', '')} {p.get('lastName', '')} ({job}): {email}")
            return "\n".join(lines) if lines else f"No contacts found in CRM matching '{q_val}'."

        elif name == "android_list_contacts":
            q_val = args.get("name_query", "").lower()
            try:
                res = run_ssh(["termux-contact-list"], capture=True)
                if "Termux:API" in res:
                    return f"Android Termux API warning: {res.strip()}. Please ensure you are using the F-Droid build of Termux and Termux:API on your phone."
                try:
                    contacts = json.loads(res)
                except json.JSONDecodeError:
                    return f"Failed to parse phone contacts JSON. Output: {res.strip()}"
                
                matches = []
                for c in contacts:
                    c_name = c.get("name", "")
                    c_num = c.get("number", "")
                    if not q_val or q_val in c_name.lower():
                        matches.append(f"- {c_name}: {c_num}")
                return "\n".join(matches[:15]) if matches else "No matching contacts found on phone."
            except Exception as e:
                return f"Failed to list phone contacts: {e}"

        elif name == "android_make_call":
            num = args["phone_number"]
            try:
                run_ssh(["termux-telephony-call", num], capture=False)
                return f"Initiated telephone call to {num}."
            except Exception as e:
                return f"Failed to place call: {e}"

        elif name == "android_set_clipboard":
            txt = args["text"]
            try:
                run_ssh(["termux-clipboard-set", txt], capture=False)
                return "Successfully updated Android phone clipboard."
            except Exception as e:
                return f"Failed to update clipboard: {e}"

        elif name == "android_get_notifications":
            try:
                res = run_ssh(["termux-notification-list"], capture=True)
                if "Termux:API" in res:
                    return f"Android Termux API warning: {res.strip()}. Please ensure you are using the F-Droid build of Termux and Termux:API on your phone."
                try:
                    notes = json.loads(res)
                except json.JSONDecodeError:
                    return f"Failed to parse phone notifications JSON. Output: {res.strip()}"
                
                lines = []
                for n in notes[:8]:
                    pkg = n.get("packageName", "").split(".")[-1]
                    title = n.get("title") or "Notification"
                    content = n.get("content") or ""
                    lines.append(f"- [{pkg}] {title}: {content}")
                return "\n".join(lines) if lines else "No active notifications on phone."
            except Exception as e:
                return f"Failed to read phone notifications: {e}"

        return f"Unknown tool: {name}"
    except Exception as e:
        print(f"[TOOL ERROR] {e}")
        return f"Error executing tool {name}: {e}"

# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>index.html not found</h3>"


@app.get("/manifest.json")
def get_manifest():
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    return FileResponse(manifest_path, media_type="application/json")


@app.get("/sw.js")
def get_sw():
    sw_path = os.path.join(os.path.dirname(__file__), "sw.js")
    return FileResponse(sw_path, media_type="application/javascript")


@app.post("/ask")
async def ask(
    audio: UploadFile = File(...),
    context: str = Form(default=""),
    history: str = Form(default="[]"),
    _: None = Depends(require_voice_token),
):
    audio_bytes = await audio.read()

    async def event_generator():
        def make_log(msg):
            print(f"[SERVER LOG] {msg}")
            return "data: " + json.dumps({"type": "log", "message": msg}) + "\n\n"

        yield make_log("💾 Saving audio query...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            yield make_log("🎙️ Transcribing audio with Whisper...")
            segments, info = whisper.transcribe(tmp_path, language="en")
            user_text = " ".join(s.text for s in segments).strip()
            
            if not user_text:
                print("[SERVER LOG] No speech detected.")
                yield "data: " + json.dumps({"type": "error", "message": "No speech detected. Please try again."}) + "\n\n"
                return

            print(f"[STT] Transcribed ({info.language}): '{user_text}'")
            yield "data: " + json.dumps({"type": "user_transcription", "message": user_text}) + "\n\n"
            yield make_log(f"✍️ Speech: '{user_text}'")

            # ── Construct Conversation Messages (with History) ──
            messages = [{"role": "system", "content": await get_system_prompt()}]
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            
            try:
                chat_history = json.loads(history)
                for msg in chat_history:
                    if msg.get("role") in ["user", "assistant"]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                print(f"[SESSION] Loaded {len(chat_history)} history messages.")
            except Exception as e:
                print(f"[SESSION ERROR] Failed to parse history: {e}")

            messages.append({"role": "user", "content": user_text})

            # Tool Calling Loop
            max_loops = 5
            reply = "I had trouble completing that task."
            
            async with httpx.AsyncClient(timeout=45) as client:
                for loop_idx in range(max_loops):
                    yield make_log(f"🤖 Querying Unified Router (loop {loop_idx + 1})...")
                    payload = {
                        "model": DEFAULT_MODEL,
                        "messages": messages,
                        "max_tokens": 300,
                        "tools": TOOLS
                    }
                    
                    r = await client.post(
                        API_URL,
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json=payload,
                    )
                    r.raise_for_status()
                    res_data = r.json()
                    choice = res_data["choices"][0]
                    message = choice["message"]
                    
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_names = [tc['function']['name'] for tc in message['tool_calls']]
                        yield make_log(f"🔧 AI requested tools: {', '.join(tool_names)}")
                        
                        messages.append(message)
                        
                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            try:
                                tool_args = clean_and_parse_json(tool_call["function"]["arguments"], tool_name)
                            except Exception as pe:
                                yield make_log(f"⚠️ Failed to parse arguments for {tool_name}: {pe}")
                                tool_result = f"Error: Failed to parse arguments as valid JSON. Details: {pe}"
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "name": tool_name,
                                    "content": tool_result
                                })
                                continue
                            
                            yield make_log(f"⚙️ Executing tool {tool_name}...")
                            
                            # Native vibration override
                            if tool_name == "vibrate_phone":
                                duration = tool_args.get("duration_ms", 500)
                                yield "data: " + json.dumps({"type": "vibrate", "duration": duration}) + "\n\n"
                                
                            tool_result = await execute_tool(tool_name, tool_args)
                            yield make_log(f"✅ Tool result: {tool_result}")
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": tool_name,
                                "content": tool_result
                            })
                        continue
                    else:
                        reply = message["content"].strip()
                        yield make_log("🤖 Final reply generated.")
                        break

            # Yield final AI reply with ElevenLabs premium audio
            print(f"[LLM] Reply: {reply}")
            
            # Split speech text from display details if a divider is present
            speech_text = reply
            for divider in ["<details>", "===", "[DISPLAY]"]:
                if divider in reply:
                    speech_text = reply.split(divider)[0].strip()
                    break
            
            yield make_log("🔊 Synthesizing premium voice with ElevenLabs...")
            base64_audio = await synthesize_speech(speech_text)
            yield "data: " + json.dumps({"type": "ai_reply", "message": reply, "audio": base64_audio}) + "\n\n"

        except httpx.HTTPError as e:
            print(f"[ERROR] HTTP connection failed: {e}")
            yield "data: " + json.dumps({"type": "error", "message": f"LLM error: {e}"}) + "\n\n"
        except Exception as e:
            print(f"[ERROR] System Exception: {e}")
            yield "data: " + json.dumps({"type": "error", "message": f"System error: {e}"}) + "\n\n"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ask-text")
async def ask_text(
    text: str = Form(...),
    context: str = Form(default=""),
    history: str = Form(default="[]"),
    _: None = Depends(require_voice_token),
):
    async def event_generator():
        def make_log(msg):
            print(f"[SERVER LOG] {msg}")
            return "data: " + json.dumps({"type": "log", "message": msg}) + "\n\n"

        try:
            user_text = text.strip()
            if not user_text:
                yield "data: " + json.dumps({"type": "error", "message": "Empty query."}) + "\n\n"
                return

            print(f"[TEXT QUERY] '{user_text}'")
            yield "data: " + json.dumps({"type": "user_transcription", "message": user_text}) + "\n\n"
            yield make_log(f"✍️ Typed: '{user_text}'")

            # ── Construct Conversation Messages (with History) ──
            messages = [{"role": "system", "content": await get_system_prompt()}]
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            
            try:
                chat_history = json.loads(history)
                for msg in chat_history:
                    if msg.get("role") in ["user", "assistant"]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                print(f"[SESSION] Loaded {len(chat_history)} history messages.")
            except Exception as e:
                print(f"[SESSION ERROR] Failed to parse history: {e}")

            messages.append({"role": "user", "content": user_text})

            # Tool Calling Loop
            max_loops = 5
            reply = "I had trouble completing that task."
            
            async with httpx.AsyncClient(timeout=45) as client:
                for loop_idx in range(max_loops):
                    yield make_log(f"🤖 Querying Unified Router (loop {loop_idx + 1})...")
                    payload = {
                        "model": DEFAULT_MODEL,
                        "messages": messages,
                        "max_tokens": 300,
                        "tools": TOOLS
                    }
                    
                    r = await client.post(
                        API_URL,
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json=payload,
                    )
                    r.raise_for_status()
                    res_data = r.json()
                    choice = res_data["choices"][0]
                    message = choice["message"]
                    
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_names = [tc['function']['name'] for tc in message['tool_calls']]
                        yield make_log(f"🔧 AI requested tools: {', '.join(tool_names)}")
                        
                        messages.append(message)
                        
                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            try:
                                tool_args = clean_and_parse_json(tool_call["function"]["arguments"], tool_name)
                            except Exception as pe:
                                yield make_log(f"⚠️ Failed to parse arguments for {tool_name}: {pe}")
                                tool_result = f"Error: Failed to parse arguments as valid JSON. Details: {pe}"
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "name": tool_name,
                                    "content": tool_result
                                })
                                continue
                            
                            yield make_log(f"⚙️ Executing tool {tool_name}...")
                            
                            # Native vibration override
                            if tool_name == "vibrate_phone":
                                duration = tool_args.get("duration_ms", 500)
                                yield "data: " + json.dumps({"type": "vibrate", "duration": duration}) + "\n\n"
                                
                            tool_result = await execute_tool(tool_name, tool_args)
                            yield make_log(f"✅ Tool result: {tool_result}")
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": tool_name,
                                "content": tool_result
                            })
                        continue
                    else:
                        reply = message["content"].strip()
                        yield make_log("🤖 Final reply generated.")
                        break

            # Yield final AI reply with ElevenLabs premium audio
            print(f"[LLM] Reply: {reply}")
            
            # Split speech text from display details if a divider is present
            speech_text = reply
            for divider in ["<details>", "===", "[DISPLAY]"]:
                if divider in reply:
                    speech_text = reply.split(divider)[0].strip()
                    break
            
            yield make_log("🔊 Synthesizing premium voice with ElevenLabs...")
            base64_audio = await synthesize_speech(speech_text)
            yield "data: " + json.dumps({"type": "ai_reply", "message": reply, "audio": base64_audio}) + "\n\n"

        except httpx.HTTPError as e:
            print(f"[ERROR] HTTP connection failed: {e}")
            yield "data: " + json.dumps({"type": "error", "message": f"LLM error: {e}"}) + "\n\n"
        except Exception as e:
            print(f"[ERROR] System Exception: {e}")
            yield "data: " + json.dumps({"type": "error", "message": f"System error: {e}"}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8765,
        ssl_keyfile="certs/cert.key",
        ssl_certfile="certs/cert.crt"
    )
