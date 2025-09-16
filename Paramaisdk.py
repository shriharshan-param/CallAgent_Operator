"""
Browser-compatible Param AI SDK converted to Python
Provides mind execution functions for Python environments
"""

import json
import time
import requests
import os
from typing import Dict, Any, Optional, Callable, List
import sseclient
from urllib.parse import urljoin
from dotenv import load_dotenv

load_dotenv()


class ParamAISDK:
    def __init__(self):
        """
        Initialize the Param AI SDK with configuration from environment variables
        """

        self.base_url = os.getenv("MINDFLOW_BASE_URL")
        self.default_mind_id = os.getenv("MIND_ID")
        self.default_share_key = os.getenv("MINDFLOW_SHARE_KEY")

        if not self.base_url or not self.default_share_key:
            raise ValueError(
                "Missing required environment variables: MINDFLOW_BASE_URL, MINDFLOW_SHARE_KEY"
            )

    def execute_mind(
        self,
        mind_name: str,
        args: Dict[str, Any],
        response_structure: Dict[str, Any],
        mind_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute mind using requests library

        Args:
            mind_name: Name of the mind to execute
            args: Arguments for the mind
            response_structure: Expected response structure
            mind_id: Optional mind ID (defaults to config value)
            session_id: Optional session ID

        Returns:
            JSON response from the API
        """
        # Prepare form data
        data = {
            "mind_name": mind_name,
            "is_default_ui_disabled": False,
            "args": json.dumps(args),
            "mind_id": mind_id or self.default_mind_id,
            "response_structure": json.dumps(response_structure),
        }

        # Only add session_id if provided
        if session_id:
            data["session_id"] = session_id

        headers = {
            "Accept": "application/json",
            "Share-Key": self.default_share_key,
            "Origin": "https://lab.paramai.studio",
            "Referer": "https://lab.paramai.studio/",
            "X-Mind-Id": mind_id or self.default_mind_id,
        }

        url = urljoin(self.base_url, "/mindflow/execute_mind")
        response = requests.post(url, data=data, headers=headers)

        if not response.ok:
            raise Exception(
                f"execute_mind failed: {response.status_code} - {response.text}"
            )

        return response.json()

    def fetch_sessions(
        self,
        mind_id: Optional[str] = None,
        share_key: Optional[str] = None,
        session_id: Optional[str] = None,
        tab_names: List[str] = [],
    ) -> Dict[str, Any]:
        """
        Get session(s) for a mind
        If session_id is provided, gets one session; otherwise, gets all sessions
        If tab_name is provided, extracts content from that specific tab

        Args:
            mind_id: Mind ID (defaults to config value)
            share_key: Share key (defaults to config value)
            session_id: Optional specific session ID
            tab_name: Optional tab name to extract content from

        Returns:
            JSON response with session data (filtered by tab if specified)
        """
        effective_mind_id = mind_id or self.default_mind_id
        effective_share_key = share_key or self.default_share_key

        if session_id:
            url = (
                f"{self.base_url}/mindflow/{effective_mind_id}/{session_id}/get_session"
            )
        else:
            url = f"{self.base_url}/mindflow/{effective_mind_id}/get_session"

        headers = {
            "Accept": "application/json",
            "share-key": effective_share_key,
            "Origin": "https://lab.paramai.studio",
            "Referer": "https://lab.paramai.studio/",
            "X-Mind-Id": effective_mind_id,
        }

        response = requests.get(url, headers=headers)

        if not response.ok:
            raise Exception(
                f"fetch_sessions failed: {response.status_code} - {response.text}"
            )

        result = response.json()

        tab_name = tab_names[0]

        for tab in tab_names:
            if tab in result.get("response", {}).get("output", {}).get("tabs", []):
                tab_name = tab
                break

        # If tab_name is provided, extract content from that specific tab
        if tab_name and result.get("response"):
            session_response = result["response"]
            output = session_response.get("output")

            if output and output.get("content"):

                # Get content from the specified tab (0th index)
                tab_content = output["content"].get(tab_name)
                if tab_content and len(tab_content) > 0:
                    # Return the content from 0th index
                    return {
                        "success": True,
                        "message": f"Content from tab '{tab_name}' fetched successfully",
                        "response": {tab_name: tab_content[0] , "logs": session_response.get("logs", []) , "args": session_response.get("args", {}), "tab_name": tab_name},
                    }
                else:
                    return {
                        "success": False,
                        "message": f"No content found in tab '{tab_name}'",
                        "response": None,
                    }
            else:
                return {
                    "success": False,
                    "message": f"Tab '{tab_name}' not found in session output",
                    "response": None,
                }

        return result

    def get_all_sessions(
        self, mind_id: Optional[str] = None, share_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all sessions for a mind

        Args:
            mind_id: Mind ID (defaults to config value)
            share_key: Share key (defaults to config value)

        Returns:
            JSON response with all sessions
        """
        return self.fetch_sessions(mind_id, share_key)

    def get_session(
        self,
        mind_id: Optional[str] = None,
        share_key: Optional[str] = None,
        session_id: Optional[str] = None,
        tab_name: List[str] = [],
    ) -> Dict[str, Any]:
        """
        Get session data using requests

        Args:
            mind_id: Mind ID (defaults to config value)
            share_key: Share key (defaults to config value)
            session_id: Specific session ID
            tab_name: Optional tab name to extract content from

        Returns:
            JSON response with session data (filtered by tab if specified)
        """
        return self.fetch_sessions(mind_id, share_key, session_id, tab_name)

    def stream_sse(
        self,
        job_id: str,
        on_event: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_complete: Optional[Callable[[Any], None]] = None,
        max_retries: int = 5,
        retry_delay: int = 5,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Stream Server-Sent Events using sseclient

        Args:
            job_id: Job ID for the stream
            on_event: Callback for each event
            on_error: Callback for errors
            on_complete: Callback for completion
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
        """
        attempts = 0

        # Replace port for SSE endpoint (5012 -> 5013)
        sse_url = self.base_url.replace(":5012", ":5013")
        url = f"{sse_url}/events/{job_id}?check=1"

        while attempts <= max_retries:
            try:
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()

                client = sseclient.SSEClient(response)

                for event in client.events():
                    try:
                        data = json.loads(event.data)

                        # Check if status is completed
                        if data and data.get("status") == "completed":
                            if on_complete:
                                on_complete(data)
                            return

                        if on_event:
                            on_event(data)

                    except json.JSONDecodeError:
                        # Handle non-JSON data
                        if on_event:
                            on_event(event.data)

                # If we reach here, stream ended normally
                return

            except Exception as error:
                attempts += 1
                if attempts <= max_retries:
                    print(
                        f"SSE connection failed (attempt {attempts}/{max_retries + 1}), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    if on_error:
                        on_error(error)
                    raise error

    def execute_mind_and_get_results(
        self,
        mind_name: str,
        is_default_ui_disabled: bool,
        args: Dict[str, Any],
        response_structure: Dict[str, Any],
        mind_id: Optional[str] = None,
        share_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute mind and get results with streaming support and polling fallback

        Args:
            mind_name: Name of the mind to execute
            is_default_ui_disabled: Whether default UI is disabled
            args: Arguments for the mind
            response_structure: Expected response structure
            mind_id: Optional mind ID
            share_key: Optional share key

        Returns:
            Final session data with results
        """
        print(f"🧠 Executing mind: {mind_name}")

        # Execute the mind first
        result = self.execute_mind(mind_name, args, response_structure, mind_id, None)
        job_id = result["job_id"]
        session_id = result["session_id"]

        print(f"Mind execution started. Job ID: {job_id}, Session ID: {session_id}")

        # Try SSE streaming first, but fall back to polling if it fails
        try:
            print("Attempting SSE streaming...")

            def on_event(data):
                if isinstance(data, dict) and data.get("message"):
                    print(f'📡 SSE: {data["message"]}')
                elif isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        if parsed and parsed.get("message"):
                            print(f'📡 SSE: {parsed["message"]}')
                        else:
                            print(f"📡 SSE: {data}")
                    except json.JSONDecodeError:
                        print(f"📡 SSE: {data}")
                else:
                    print(f"📡 SSE: {data}")

            def on_error(err):
                print(f"SSE error (will fall back to polling): {err}")

            self.stream_sse(
                job_id,
                on_event=on_event,
                on_error=on_error,
                max_retries=2,
                retry_delay=3,
            )

        except Exception as sse_error:
            print(f"SSE streaming failed, using polling fallback: {sse_error}")

            # Polling fallback - check session status periodically
            max_polls = 20  # Maximum 2 minutes of polling (6s * 20)
            poll_interval = 6  # 6 seconds

            for i in range(max_polls):
                print(f"📊 Polling attempt {i + 1}/{max_polls}...")

                try:
                    session_data = self.get_session(mind_id, share_key, session_id)

                    # Check if execution is complete
                    if (
                        session_data
                        and session_data.get("response")
                        and session_data.get("execution_status") == "completed"
                    ):
                        print("✅ Mind execution completed via polling")
                        return session_data

                    # Wait before next poll
                    if i < max_polls - 1:
                        time.sleep(poll_interval)

                except Exception as poll_error:
                    print(f"Polling attempt {i + 1} failed: {poll_error}")

            print("⏰ Polling timeout reached, fetching final session data...")

        # Fetch final session data
        print("Fetching final session data...")
        return self.get_session(mind_id, share_key, session_id)