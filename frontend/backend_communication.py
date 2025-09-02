#!/usr/bin/env python3
"""
KWE CLI Backend Communication Module

This module handles all HTTP communication with the backend server:
- Health checks and connectivity verification
- Chat request handling
- Code generation requests
- Code analysis requests
- Context management API calls
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx


class BackendCommunication:
    """Handles all HTTP communication with the KWE CLI backend server."""
    
    def __init__(self, backend_url: str = None, timeout: float = 30.0):
        """
        Initialize backend communication.
        
        Args:
            backend_url: The base URL of the backend server (auto-detected if None)
            timeout: Request timeout in seconds
        """
        if backend_url is None:
            import os
            backend_host = os.environ.get("KWE_BACKEND_HOST", "localhost")
            backend_port = os.environ.get("KWE_BACKEND_PORT", "8000")
            backend_url = f"http://{backend_host}:{backend_port}"
        
        self.backend_url = backend_url
        self.http_client = httpx.AsyncClient(timeout=timeout)
    
    async def check_health(self) -> Dict[str, Any]:
        """Check if the backend server is running and healthy."""
        try:
            response = await self.http_client.get(f"{self.backend_url}/health")
            # Return the JSON status for compatibility with frontend tests
            data = response.json()
            if isinstance(data, dict) and "status" in data:
                return data
            # Fallback if health endpoint does not return JSON
            if response.status_code == 200:
                return {"status": "healthy"}
            return {"status": "unhealthy", "code": response.status_code}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_chat(self) -> bool:
        """Check if the chat endpoint is reachable by sending a ping."""
        try:
            result = await self.send_chat_request("ping")
            # Success if no error indicators
            if isinstance(result, str) and not result.lower().startswith(("error", "connection error")):
                return True
            return False
        except Exception:
            return False
    
    async def send_request(self, endpoint: str, data: Dict[str, Any] = None, method: str = "POST") -> Dict[str, Any]:
        """
        Send a generic HTTP request to the backend.
        """
        try:
            url = f"{self.backend_url}{endpoint}"
            if method.upper() == "GET":
                response = await self.http_client.get(url)
            elif method.upper() == "POST":
                response = await self.http_client.post(url, json=data)
            elif method.upper() == "DELETE":
                response = await self.http_client.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code in (200, 201, 204):
                return response.json() if response.content else {"success": True}
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": f"Request failed: {e}"}

    async def send_chat_request(self, prompt: str) -> str:
        """
        Send a chat request to the backend.
        """
        try:
            payload = {
                "prompt": prompt,
                "context": "cli_interaction",
                "timestamp": datetime.now().isoformat()
            }
            response = await self.http_client.post(
                f"{self.backend_url}/api/chat", json=payload
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response received")
            return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Connection error: {e}"

    async def send_code_generation_request(
        self,
        description: str,
        language: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a code generation request to the backend.
        """
        try:
            payload = {"description": description, "language": language, "context": context}
            response = await self.http_client.post(
                f"{self.backend_url}/api/generate", json=payload
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {e}"}

    async def send_code_analysis_request(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a code analysis request to the backend.
        """
        try:
            payload = {"file_path": file_path, "language": language}
            response = await self.http_client.post(
                f"{self.backend_url}/api/analyze", json=payload
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {e}"}

    async def send_context_request(
        self,
        content: str,
        tags: List[str],
        priority: str
    ) -> Dict[str, Any]:
        """
        Send a context storage request to the backend.
        """
        try:
            payload = {"content": content, "tags": tags, "priority": priority}
            response = await self.http_client.post(
                f"{self.backend_url}/api/context", json=payload
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {e}"}

    async def search_context_request(self, query: str, limit: int) -> Dict[str, Any]:
        """
        Send a context search request to the backend.
        """
        try:
            params = {"query": query, "limit": limit}
            response = await self.http_client.get(
                f"{self.backend_url}/api/context/search", params=params
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {e}"}

    async def list_context_request(self) -> Dict[str, Any]:
        """
        List all stored context entries.
        """
        try:
            response = await self.http_client.get(f"{self.backend_url}/api/context")
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {e}"}

    async def delete_context_request(self, context_id: str) -> Dict[str, Any]:
        """
        Delete a specific context entry.
        """
        try:
            response = await self.http_client.delete(
                f"{self.backend_url}/api/context/{context_id}"
            )
            if response.status_code in (200, 201, 204):
                return response.json() if response.content else {"success": True}
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {e}"}

    async def get_backend_status(self) -> Dict[str, Any]:
        """
        Get comprehensive backend status information.
        """
        try:
            response = await self.http_client.get(f"{self.backend_url}/api/status")
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Connection error: {e}"}

    async def cleanup(self) -> None:
        """Clean up HTTP client resources."""
        await self.http_client.aclose()

    def get_backend_url(self) -> str:
        """Get the current backend URL."""
        return self.backend_url

    def update_backend_url(self, new_url: str) -> None:
        """Update the backend URL."""
        self.backend_url = new_url

    async def stream_generate(self, prompt: str):
        """Stream SSE events for the generate pipeline."""
        url = f"{self.backend_url}/api/stream/generate"
        try:
            async with self.http_client.stream("POST", url, json={"prompt": prompt}) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    if raw_line.startswith("event: "):
                        evt = raw_line.split(": ", 1)[1].strip()
                        data_line = await resp.aiter_lines().__anext__()
                        if data_line.startswith("data: "):
                            data = data_line.split(": ", 1)[1]
                        else:
                            data = ""
                        yield {"event": evt, "data": data}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    async def run_code_quality(self, paths: Optional[List[str]] = None, check_only: bool = True) -> Dict[str, Any]:
        url = f"{self.backend_url}/api/dev/code_quality"
        try:
            resp = await self.http_client.post(
                url, json={"paths": paths or ["."], "check_only": bool(check_only)}
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_tests(self, marker: Optional[str] = None, path: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.backend_url}/api/dev/test"
        payload: Dict[str, Any] = {}
        if marker:
            payload["marker"] = marker
        if path:
            payload["path"] = path
        try:
            resp = await self.http_client.post(url, json=payload)
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def read_file(self, path: str) -> Dict[str, Any]:
        url = f"{self.backend_url}/api/file/read"
        try:
            resp = await self.http_client.get(url, params={"path": path})
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def diff_patch(self, file_path: str, content: str) -> Dict[str, Any]:
        url = f"{self.backend_url}/api/patch/diff"
        try:
            resp = await self.http_client.post(url, json={"file_path": file_path, "content": content})
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def apply_patch(self, file_path: str, content: str, create_dirs: bool = False) -> Dict[str, Any]:
        url = f"{self.backend_url}/api/patch/apply"
        try:
            resp = await self.http_client.post(
                url, json={"file_path": file_path, "content": content, "create_dirs": create_dirs}
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_config(self) -> Dict[str, Any]:
        try:
            resp = await self.http_client.get(f"{self.backend_url}/api/config/get")
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def update_config(self, **kwargs: Any) -> Dict[str, Any]:
        """Update backend configuration via API."""
        try:
            resp = await self.http_client.post(f"{self.backend_url}/api/config/update", json=kwargs)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_models(self) -> Dict[str, Any]:
        """Fetch available Ollama models and related info."""
        try:
            resp = await self.http_client.get(f"{self.backend_url}/api/models")
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_available_models(self) -> Dict[str, Any]:
        """Alias for get_models to support frontend tests."""
        return await self.get_models()

    async def get_context(self) -> Dict[str, Any]:
        """Alias for list_context_request to support frontend tests."""
        return await self.list_context_request()

    async def session_list(self) -> Dict[str, Any]:
        try:
            resp = await self.http_client.get(f"{self.backend_url}/api/session/list")
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def session_create(self, name: Optional[str] = None) -> Dict[str, Any]:
        try:
            resp = await self.http_client.post(f"{self.backend_url}/api/session/create", json={"name": name})
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


    async def session_select(self, session_id: str) -> Dict[str, Any]:
        """Alias for select session to support frontend coordination."""
        try:
            resp = await self.http_client.post(
                f"{self.backend_url}/api/session/select",
                json={"session_id": session_id}
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def session_history(self, limit: int = 100) -> Dict[str, Any]:
        """Get session history from backend."""
        try:
            resp = await self.http_client.get(
                f"{self.backend_url}/api/session/history",
                params={"limit": limit}
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def session_save_markdown(self) -> Dict[str, Any]:
        """Save current session to markdown format."""
        try:
            resp = await self.http_client.post(f"{self.backend_url}/api/session/save_markdown")
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def generate_code_with_tests(self, description: str, language: str, 
                                     code_path: str, test_path: str) -> Dict[str, Any]:
        """Generate code with tests through backend."""
        try:
            payload = {
                "description": description,
                "language": language,
                "code_path": code_path,
                "test_path": test_path
            }
            resp = await self.http_client.post(
                f"{self.backend_url}/api/generate/code_with_tests",
                json=payload
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
