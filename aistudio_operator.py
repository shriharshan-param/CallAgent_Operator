import asyncio
import audioop
import base64
import io
import json
import os
import time
import uuid
import wave
import tempfile
from .config import Config

import numpy as np
import pandas as pd
import requests
import websockets
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

load_dotenv()

class VoiceGatewayApp:
    def __init__(self):
        self.twilio_client = Client(
            Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN
        )

        # Store active connections
        self.active_connections = {}

        # Global metadata for calls (to store from_number, excel_path, df per call_sid)
        self.call_metadata = {}

        # Store pending uploads for outbound calls (call_sid -> excel_path, df)
        self.pending_uploads_dir = tempfile.gettempdir()

    def _get_upload_file_path(self, upload_id):
        """Get file path for upload data"""
        return os.path.join(self.pending_uploads_dir, f"upload_{upload_id}.json")

    def _store_upload_data(self, upload_id, data):
        """Store upload data to file"""
        file_path = self._get_upload_file_path(upload_id)
        with open(file_path, 'w') as f:
            # Convert DataFrame to dict for JSON serialization
            data_copy = data.copy()
            if 'df' in data_copy:
                data_copy['df'] = data_copy['df'].to_dict('records')
            json.dump(data_copy, f)

    def _get_upload_data(self, upload_id):
        """Retrieve upload data from file"""
        file_path = self._get_upload_file_path(upload_id)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Convert dict back to DataFrame
                if 'df' in data:
                    data['df'] = pd.DataFrame(data['df'])
                return data
        return None

    def _remove_upload_data(self, upload_id):
        """Remove upload data file"""
        file_path = self._get_upload_file_path(upload_id)
        if os.path.exists(file_path):
            os.remove(file_path)

    def _list_upload_ids(self):
        """List all available upload IDs"""
        upload_files = [f for f in os.listdir(self.pending_uploads_dir) if f.startswith('upload_') and f.endswith('.json')]
        return [f.replace('upload_', '').replace('.json', '') for f in upload_files]

    def get_existing_prompts(self):
        """Get list of all existing prompts across all pages"""
        all_prompts = []
        page_number = 0
        page_size = 50
        try:
            while True:
                url = f"https://api.hume.ai/v0/evi/prompts?page_number={page_number}&page_size={page_size}"
                headers = {"X-Hume-Api-Key": Config.HUME_API_KEY}

                # Set a reasonable timeout for the request
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    raise requests.exceptions.HTTPError(
                        f"Failed to get prompts page {page_number}: {response.text}",
                        response=response,
                    )

                data = response.json()
                all_prompts.extend(data.get("prompts_page", []))

                total_pages = data.get("total_pages", 1)
                if page_number + 1 >= total_pages:
                    break

                page_number += 1

            return {"prompts_page": all_prompts}
        except Exception as e:
            raise RuntimeError(f"Unexpected error in get_existing_prompts: {str(e)}")

    def get_existing_configs(self):
        """Get list of all existing configs across all pages"""
        all_configs = []
        page_number = 0
        page_size = 50

        try:
            while True:
                url = f"https://api.hume.ai/v0/evi/configs?page_number={page_number}&page_size={page_size}"
                headers = {"X-Hume-Api-Key": Config.HUME_API_KEY}
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    raise requests.exceptions.HTTPError(
                        f"Failed to get configs page {page_number}: {response.text}",
                        response=response,
                    )

                data = response.json()
                all_configs.extend(data.get("configs_page", []))

                total_pages = data.get("total_pages", 1)
                if page_number + 1 >= total_pages:
                    break

                page_number += 1

            return {"configs_page": all_configs}
        except Exception as e:
            raise RuntimeError(f"Unexpected error in get_existing_configs: {str(e)}")

    def get_existing_tools(self):
        """Get list of all existing tools across all pages"""
        all_tools = []
        page_number = 0
        page_size = 50

        try:
            while True:
                url = f"https://api.hume.ai/v0/evi/tools?page_number={page_number}&page_size={page_size}"
                headers = {"X-Hume-Api-Key": Config.HUME_API_KEY}

                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    raise requests.exceptions.HTTPError(
                        f"Failed to get tools page {page_number}: {response.text}"
                    )

                data = response.json()
                all_tools.extend(data.get("tools_page", []))

                total_pages = data.get("total_pages", 1)
                if page_number + 1 >= total_pages:
                    break

                page_number += 1

            return {"tools_page": all_tools}
        except Exception as e:
            raise RuntimeError(f"Error in get_existing_tools: {str(e)}")

    def create_hume_prompt(self, prompt_text: str):
        """Create a prompt in Hume AI and return the prompt ID"""
        url = "https://api.hume.ai/v0/evi/prompts"
        headers = {
            "X-Hume-Api-Key": Config.HUME_API_KEY,
            "Content-Type": "application/json",
        }
        timestamp = int(time.time())
        prompt_data = {
            "name": f"Dynamic Prompt {timestamp}",
            "text": prompt_text,
        }

        try:
            response = requests.post(url, headers=headers, json=prompt_data, timeout=10)

            if response.status_code != 201:
                raise requests.exceptions.HTTPError(
                    f"Failed to create prompt: Status {response.status_code}, Response: {response.text}"
                )

            prompt_info = response.json()
            prompt_id = prompt_info.get("id")

            if not prompt_id:
                raise ValueError("Response does not contain prompt ID")

            return prompt_id

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.HTTPError(f"Network error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error in create_hume_prompt: {str(e)}")

    def create_excel_tool(self):
        """Create Excel query tool and return tool ID and version"""
        try:
            existing_tools = self.get_existing_tools()
            if existing_tools and "tools_page" in existing_tools:
                for tool in existing_tools["tools_page"]:
                    if tool.get("name") == "get_excel_data":
                        return tool["id"], tool["version"]
            url = "https://api.hume.ai/v0/evi/tools"
            headers = {
                "X-Hume-Api-Key": Config.HUME_API_KEY,
                "Content-Type": "application/json",
            }
            tool_data = {
                "name": "get_excel_data",
                "parameters": '{ "type": "object", "properties": { "query": { "type": "string", "description": "Natural language query about the Excel data, e.g. What are my recent transactions" } }, "required": ["query"] }',
                "version_description": "Queries customer Excel data using natural language.",
                "description": "This tool queries the customer's uploaded Excel data to retrieve information.",
                "fallback_content": "Unable to access your data right now.",
            }
            response = requests.post(url, headers=headers, json=tool_data)
            if response.status_code == 201:
                tool_info = response.json()
                return tool_info["id"], tool_info["version"]
            else:
                raise requests.exceptions.HTTPError(
                    f"Failed to create excel tool: Status {response.status_code}, Response: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.HTTPError(
                f"Network error creating excel tool: {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(f"Error in create_excel_tool: {str(e)}")

    def create_dynamic_hume_config(self, prompt_text: str, customer_phone: str):
        """Create a configuration dynamically for each customer"""
        try:
            # First create the prompt
            prompt_id = self.create_hume_prompt(prompt_text)
            if not prompt_id:
                raise RuntimeError("Failed to create Hume prompt")

            # Get or create tools
            excel_tool_id, excel_tool_version = self.create_excel_tool()

            # Build tools array
            tools = []
            if excel_tool_id:
                tools.append({"id": excel_tool_id, "version": excel_tool_version})

            url = "https://api.hume.ai/v0/evi/configs"
            headers = {
                "X-Hume-Api-Key": Config.HUME_API_KEY,
                "Content-Type": "application/json",
            }
            timestamp = int(time.time())
            config_data = {
                "evi_version": "3",
                "name": f"Customer Config {customer_phone[-4:]} {timestamp}",
                "prompt": {"id": prompt_id, "version": 0},
                "voice": {"provider": "HUME_AI", "name": "ITO"},
                "language_model": {
                    "model_provider": "ANTHROPIC",
                    "model_resource": "claude-3-5-sonnet-latest",
                    "temperature": 0.7,
                },
                "event_messages": {
                    "on_new_chat": {
                        "enabled": True,
                        "text": "Hello! I'm calling from Axis Mutual Funds customer service, May I know what Issue you were facing with the application",
                    },
                    "on_inactivity_timeout": {"enabled": False, "text": ""},
                    "on_max_duration_timeout": {"enabled": False, "text": ""},
                },
                "tools": tools,
                "builtin_tools": [],
            }
            response = requests.post(url, headers=headers, json=config_data)
            if response.status_code == 201:
                config_info = response.json()
                return config_info["id"]
            else:
                raise requests.exceptions.HTTPError(
                    f"Failed to create Hume config: Status {response.status_code}, Response: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.HTTPError(
                f"Network error creating Hume config: {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(f"Error in create_dynamic_hume_config: {str(e)}")

    async def make_outbound_call(self, phone_number, file_path, prompt):
        """Initiate an outbound call with Excel file and dynamic config"""
        try:
            # Load DataFrame
            df = pd.read_excel(file_path)

            # Create dynamic config for this customer
            dynamic_config_id = self.create_dynamic_hume_config(prompt, phone_number)
            if not dynamic_config_id:
                return {
                    "success": False,
                    "error": "Failed to create configuration for customer",
                }

            # Store in pending_uploads_dir
            upload_id = str(uuid.uuid4())
            upload_data = {
                "phone_number": phone_number,
                "excel_path": file_path,
                "df": df,
                "config_id": dynamic_config_id,
            }
            self._store_upload_data(upload_id, upload_data)

            # Make the call
            twilio_number = Config.TWILIO_PHONE_NUMBER
            base_url = Config.SERVER_BASE_URL
            if not twilio_number or not base_url:
                return {
                    "success": False,
                    "error": "TWILIO_PHONE_NUMBER or BASE_URL not set",
                }

            call = self.twilio_client.calls.create(
                to=phone_number,
                from_=twilio_number,
                url=f"{base_url}/twilio/voice?upload_id={upload_id}",
            )

            return {
                "success": True,
                "call_sid": call.sid,
                "upload_id": upload_id,
                "config_id": dynamic_config_id,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def twilio_voice_webhook(self, request):
        """Handle incoming calls from Twilio"""
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")
        direction = form_data.get("Direction", "inbound")

        # Get upload_id from query parameters for outbound calls
        upload_id = request.query_params.get("upload_id")

        # For outbound calls, get the Excel from pending_outbound_uploads
        if (
            direction == "outbound-api"
            and upload_id
        ):
            excel_data = self._get_upload_data(upload_id)
            if excel_data:
                excel_path = excel_data["excel_path"]
                df = excel_data["df"]
                dynamic_config_id = excel_data["config_id"]
                self._remove_upload_data(upload_id)
        else:
            # For inbound calls, find the most recent upload for this from_number
            excel_path = None
            df = None
            dynamic_config_id = None
            latest_upload_time = 0
            latest_upload_id = None

            # Look through call_metadata for pending inbound uploads
            for key, metadata in list(self.call_metadata.items()):
                if (
                    metadata.get("pending_inbound")
                    and metadata.get("phone_number") == from_number
                ):
                    upload_time = os.path.getmtime(metadata.get("excel_path"))
                    if upload_time > latest_upload_time:
                        latest_upload_time = upload_time
                        excel_path = metadata.get("excel_path")
                        df = metadata.get("df")
                        dynamic_config_id = metadata.get("config_id")
                        latest_upload_id = key

            # Remove the pending flag and associate with the call
            if latest_upload_id:
                self.call_metadata[latest_upload_id]["pending_inbound"] = False
                self.call_metadata[latest_upload_id]["call_sid"] = call_sid

        # Store in call_metadata
        self.call_metadata[call_sid] = {
            "from_number": from_number,
            "excel_path": excel_path,
            "df": df,
            "config_id": dynamic_config_id,
            "direction": direction,
        }


        # Create TwiML response to start streaming
        response = VoiceResponse()

        # Connect to our WebSocket endpoint
        base_url = Config.SERVER_BASE_URL
        if base_url is None:
            raise RuntimeError("BASE_URL environment variable not set")

        ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
        stream_url = f"{ws_base}/hume/ws/{call_sid}"


        connect = response.connect()
        connect.stream(url=stream_url, name="HumeAI Stream")

        # return str(response)
        twiml_response = str(response)
        return twiml_response

    async def connect_to_hume_ai_dynamic(self, config_id: str):
        """Establish WebSocket connection to Hume AI with dynamic config"""
        try:
            api_key = Config.HUME_API_KEY
            hume_ws_url = f"wss://api.hume.ai/v0/evi/chat?api_key={api_key}&config_id={config_id}"
            websocket = await websockets.connect(hume_ws_url)
            return websocket
        except Exception as e:
            raise RuntimeError(f"Error connecting to Hume AI: {str(e)}")

    async def hume_websocket_endpoint(self, websocket, call_sid: str):
        """WebSocket endpoint that bridges Twilio and Hume AI"""
        try:
            await websocket.accept()

            metadata = self.call_metadata.get(call_sid, {})
            from_number = metadata.get("from_number")
            dynamic_config_id = metadata.get("config_id")  # Get the dynamic config ID

            if not from_number or not dynamic_config_id:
                raise ValueError(f"Missing data for call {call_sid}")

            # Get pre-loaded customer_data
            customer_data = metadata.get("df")

            # Connect with dynamic config
            hume_ws = await self.connect_to_hume_ai_dynamic(dynamic_config_id)
            if not hume_ws:
                raise RuntimeError("Failed to connect to Hume AI")

            self.active_connections[call_sid] = {
                "twilio_ws": websocket,
                "hume_ws": hume_ws,
                "stream_sid": None,
                "customer_data": customer_data,  # Store per-call DataFrame
            }

            session_settings = {
                "type": "session_settings",
                "audio": {"encoding": "linear16", "sample_rate": 8000, "channels": 1},
            }
            await hume_ws.send(json.dumps(session_settings))

            await asyncio.gather(
                self.handle_twilio_messages(websocket, hume_ws, call_sid),
                self.handle_hume_messages(hume_ws, websocket, call_sid),
                return_exceptions=True,
            )
        except Exception as e:
            raise RuntimeError(f"WebSocket endpoint error: {str(e)}")
        finally:
            await self.cleanup_connections(call_sid)

    async def handle_twilio_messages(self, twilio_ws, hume_ws, call_sid):
        """Handle messages from Twilio and forward to Hume AI"""
        try:
            while True:
                try:
                    message = await twilio_ws.receive_text()
                    data = json.loads(message)
                    event_type = data.get("event")
                    if event_type == "media":
                        mulaw_b64 = data["media"]["payload"]
                        mulaw_bytes = base64.b64decode(mulaw_b64)
                        pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
                        pcm_b64 = base64.b64encode(pcm_bytes).decode("utf-8")
                        hume_message = {"type": "audio_input", "data": pcm_b64}
                        await hume_ws.send(json.dumps(hume_message))
                    elif event_type == "start":
                        stream_sid = data.get("streamSid")
                        if call_sid in self.active_connections:
                            self.active_connections[call_sid]["stream_sid"] = stream_sid
                    elif event_type == "stop":
                        break
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid JSON from Twilio: {str(e)}")
                except Exception as e:
                    raise RuntimeError(f"Error processing Twilio message: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error in Twilio message handler: {str(e)}")

    def extract_pcm_from_wav(self, wav_data):
        """Extract PCM data from WAV file bytes"""
        try:
            wav_io = io.BytesIO(wav_data)
            with wave.open(wav_io, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                pcm_data = wav_file.readframes(wav_file.getnframes())
                return pcm_data, channels, sample_width, framerate
        except Exception as e:
            raise RuntimeError(f"Error extracting PCM from WAV: {str(e)}")

    async def handle_tool_call(self, tool_name: str, parameters: dict, call_sid: str):
        """
        Action Handler: Execute tool and return JSON result.
        """
        try:
            if tool_name == "get_excel_data":
                query = parameters.get("query")
                if not query:
                    return {"error": "Missing required parameter: query"}
                customer_data = self.active_connections.get(call_sid, {}).get(
                    "customer_data"
                )
                if customer_data is None:
                    return {
                        "error": "No customer data Excel file has been uploaded or found for your account. Please upload one first."
                    }
                try:
                    return await self.query_excel_data(call_sid, query)
                except Exception as e:
                    return {"error": f"Failed to query Excel data: {str(e)}"}
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    async def query_excel_data(self, call_sid: str, query: str) -> dict:
        """Query customer data based on natural language query"""
        customer_data = self.active_connections.get(call_sid, {}).get("customer_data")
        if customer_data is None or len(customer_data) == 0:
            return {"error": "No customer data found for your account"}
        query_lower = query.lower()
        if any(
            word in query_lower for word in ["transaction", "recent", "last", "latest"]
        ):
            column_types = self.detect_column_types(customer_data)
            result_df = customer_data.copy()
            if column_types["date"]:
                date_col = column_types["date"][0]
                result_df[date_col] = pd.to_datetime(
                    result_df[date_col], errors="coerce"
                )
                result_df = result_df.sort_values(date_col, ascending=False)
            result_df = result_df.head(10)
            if len(result_df) == 0:
                return {"message": "No transaction records found for your account"}
            data = result_df.to_dict(orient="records")
            return {"message": "Here are your recent transactions", "data": data}
        elif any(word in query_lower for word in ["balance", "total"]):
            column_types = self.detect_column_types(customer_data)
            if column_types["balance"] or column_types["amount"]:
                balance_col = (
                    column_types["balance"][0]
                    if column_types["balance"]
                    else column_types["amount"][0]
                )
                current_balance = (
                    customer_data[balance_col].iloc[-1]
                    if not customer_data.empty
                    else 0
                )
                return {
                    "message": f"Your current balance is {current_balance}",
                    "balance": current_balance,
                }
            return {"message": "Balance information not found in your account data"}
        else:
            return {
                "message": f"Your account has {len(customer_data)} records",
                "columns": list(customer_data.columns),
                "sample": customer_data.head(3).to_dict(orient="records"),
            }

    def detect_column_types(self, df: pd.DataFrame) -> dict:
        """Detect column types similar to reference"""
        column_types = {
            "date": [],
            "amount": [],
            "balance": [],
        }
        for col in df.columns:
            col_lower = col.lower()
            if "date" in col_lower or "time" in col_lower:
                column_types["date"].append(col)
            elif "amount" in col_lower or "value" in col_lower:
                column_types["amount"].append(col)
            elif "balance" in col_lower or "total" in col_lower:
                column_types["balance"].append(col)
        return column_types

    async def handle_hume_messages(self, hume_ws, twilio_ws, call_sid):
        """Handle messages from Hume AI and forward to Twilio"""
        try:
            while True:
                try:
                    message = await hume_ws.recv()
                    data = json.loads(message)
                    message_type = data.get("type")
                    if message_type == "audio_output":
                        audio_data = data.get("data")
                        if audio_data:
                            try:
                                wav_bytes = base64.b64decode(audio_data)
                                pcm_data, channels, sample_width, framerate = (
                                    self.extract_pcm_from_wav(wav_bytes)
                                )
                                if (
                                    pcm_data
                                    and framerate is not None
                                    and channels is not None
                                ):
                                    if framerate != 8000:
                                        ratio = framerate // 8000
                                        if ratio > 1:
                                            pcm_array = np.frombuffer(
                                                pcm_data, dtype=np.int16
                                            )
                                            downsampled = pcm_array[::ratio]
                                            pcm_data = downsampled.tobytes()
                                    if channels == 2:
                                        stereo_data = np.frombuffer(
                                            pcm_data, dtype=np.int16
                                        )
                                        mono_data = (
                                            (stereo_data[0::2] + stereo_data[1::2]) / 2
                                        ).astype(np.int16)
                                        pcm_data = mono_data.tobytes()
                                    mulaw_bytes = audioop.lin2ulaw(pcm_data, 2)
                                    mulaw_b64 = base64.b64encode(mulaw_bytes).decode(
                                        "utf-8"
                                    )
                                    stream_sid = self.active_connections.get(
                                        call_sid, {}
                                    ).get("stream_sid")
                                    if stream_sid:
                                        twilio_message = {
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {"payload": mulaw_b64},
                                        }
                                        await twilio_ws.send_text(
                                            json.dumps(twilio_message)
                                        )
                            except Exception as e:
                                raise RuntimeError(f"Error processing audio output: {str(e)}")
                    elif message_type == "tool_call":
                        tool_call_id = data.get("tool_call_id")
                        tool_name = data.get("name")
                        parameters_str = data.get("parameters")
                        response_required = data.get("response_required", True)
                        parameters = {}
                        if parameters_str:
                            try:
                                parameters = json.loads(parameters_str)
                            except json.JSONDecodeError as e:
                                raise RuntimeError(f"Failed to parse parameters: {str(e)}")
                                parameters = {}
                        tool_result = await self.handle_tool_call(
                            tool_name, parameters, call_sid
                        )
                        if response_required:
                            tool_response = {
                                "type": "tool_response",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(tool_result),
                                "tool_name": tool_name,
                                "tool_type": "function",
                            }
                            await hume_ws.send(json.dumps(tool_response))
                        else:
                            pass  # Tool call does not require response
                    elif message_type == "error":
                        raise RuntimeError(f"Hume error: {data}")
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid JSON from Hume: {str(e)}")
                except Exception as e:
                    raise RuntimeError(f"Error processing Hume message: {str(e)}")
        except websockets.exceptions.ConnectionClosed:
            pass  # Connection closure handled by upper layer
        except Exception as e:
            raise RuntimeError(f"Error in Hume message handler: {str(e)}")

    async def cleanup_connections(self, call_sid: str):
        """Clean up WebSocket connections and temporary files"""
        # Store phone number before deleting metadata
        phone_number = None
        if call_sid in self.call_metadata:
            phone_number = self.call_metadata[call_sid].get("from_number")
        
        if call_sid in self.active_connections:
            connection_info = self.active_connections[call_sid]
            # Close Hume WebSocket
            try:
                if connection_info.get("hume_ws"):
                    await connection_info["hume_ws"].close()
            except Exception:
                pass  # Ignore cleanup errors to avoid masking original errors
            # Remove from active connections
            del self.active_connections[call_sid]

        # Clean up call metadata and temporary file
        if call_sid in self.call_metadata:
            excel_path = self.call_metadata[call_sid].get("excel_path")
            if excel_path and os.path.exists(excel_path):
                try:
                    os.remove(excel_path)
                except Exception:
                    pass  # Ignore cleanup errors
            del self.call_metadata[call_sid]

        # Clean up any remaining upload files that weren't used
        if phone_number:
            try:
                upload_ids = self._list_upload_ids()
                for upload_id in upload_ids:
                    upload_data = self._get_upload_data(upload_id)
                    if upload_data and upload_data.get("phone_number") == phone_number:
                        # Remove the upload file
                        self._remove_upload_data(upload_id)
                        # Also clean up the associated Excel file if it exists
                        excel_path = upload_data.get("excel_path")
                        if excel_path and os.path.exists(excel_path):
                            try:
                                os.remove(excel_path)
                            except Exception:
                                pass  # Ignore cleanup errors
            except Exception:
                pass  # Ignore cleanup errors to avoid masking original errors


# Create an instance of the VoiceGatewayApp
voice_gateway = VoiceGatewayApp()


# Expose the make_outbound_call function for direct use
async def make_outbound_call(phone_number, file_path, prompt):
    """
    Initiate an outbound call with Excel file and dynamic config.
    This function can be called directly from other modules.
    """
    return await voice_gateway.make_outbound_call(phone_number, file_path, prompt)