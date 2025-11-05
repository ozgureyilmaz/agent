#!/usr/bin/env python3

import os
import sys
import json
from typing import Dict, Any, Tuple, Optional, List, cast
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import ToolParam, MessageParam, TextBlockParam, ToolResultBlockParam

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    GENAI_AVAILABLE = False


class FileTools:
    @staticmethod
    def read_file(filepath: str) -> Dict[str, Any]:
        try:
            path = Path(filepath).resolve()
            if not path.exists():
                return {"error": f"File not found: {filepath}"}
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return {
                "success": True,
                "filepath": str(path),
                "content": content,
                "size": len(content)
            }
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}
    
    @staticmethod
    def list_files(directory: str = ".") -> Dict[str, Any]:
        try:
            path = Path(directory).resolve()
            if not path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            if not path.is_dir():
                return {"error": f"Not a directory: {directory}"}
            
            files = []
            dirs = []
            
            for item in path.iterdir():
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "size": item.stat().st_size,
                        "path": str(item)
                    })
                elif item.is_dir():
                    dirs.append({
                        "name": item.name,
                        "path": str(item)
                    })
            
            files.sort(key=lambda x: x['name'])
            dirs.sort(key=lambda x: x['name'])
            
            return {
                "success": True,
                "directory": str(path),
                "directories": dirs,
                "files": files,
                "total_files": len(files),
                "total_dirs": len(dirs)
            }
        except Exception as e:
            return {"error": f"Error listing directory: {str(e)}"}
    
    @staticmethod
    def create_file(filepath: str, content: str) -> Dict[str, Any]:
        try:
            path = Path(filepath).resolve()

            if path.exists():
                return {"error": f"File already exists: {filepath}"}

            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "filepath": str(path),
                "size": len(content),
                "message": f"Created file: {filepath}"
            }
        except Exception as e:
            return {"error": f"Error creating file: {str(e)}"}
    
    @staticmethod
    def edit_file(filepath: str, old_str: str, new_str: str) -> Dict[str, Any]:
        try:
            path = Path(filepath).resolve()
            
            if not path.exists():
                return {"error": f"File not found: {filepath}"}
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_str not in content:
                return {"error": f"String not found in file: '{old_str}'"}

            count = content.count(old_str)
            new_content = content.replace(old_str, new_str)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                "success": True,
                "filepath": str(path),
                "replacements": count,
                "message": f"Replaced {count} occurrence(s) of '{old_str}' with '{new_str}'"
            }
        except Exception as e:
            return {"error": f"Error editing file: {str(e)}"}


class AIAgent:
    def __init__(self):
        load_dotenv()

        self.provider: Optional[str] = None
        self.client: Optional[Anthropic] = None
        self.model: Any = None
        self.tools = FileTools()
        self.max_iterations = 10

        if os.getenv("GEMINI_API_KEY"):
            self.init_gemini()
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.init_anthropic()
        else:
            raise ValueError("No API key found! Please set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env file")
    
    def init_gemini(self):
        if not GENAI_AVAILABLE or genai is None:
            raise ImportError("google-generativeai package not installed")

        self.provider = "gemini"
        configure_func = getattr(genai, 'configure')
        configure_func(api_key=os.getenv("GEMINI_API_KEY"))

        tools = [
            {
                "function_declarations": [
                    {
                        "name": "read_file",
                        "description": "Read the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {
                                    "type": "string",
                                    "description": "Path to the file to read"
                                }
                            },
                            "required": ["filepath"]
                        }
                    },
                    {
                        "name": "list_files",
                        "description": "List files and directories in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "directory": {
                                    "type": "string",
                                    "description": "Directory path (default: current directory)"
                                }
                            }
                        }
                    },
                    {
                        "name": "create_file",
                        "description": "Create a new file with content",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {
                                    "type": "string",
                                    "description": "Path for the new file"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to write to the file"
                                }
                            },
                            "required": ["filepath", "content"]
                        }
                    },
                    {
                        "name": "edit_file",
                        "description": "Edit a file by replacing exact string matches",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filepath": {
                                    "type": "string",
                                    "description": "Path to the file to edit"
                                },
                                "old_str": {
                                    "type": "string",
                                    "description": "Exact string to replace"
                                },
                                "new_str": {
                                    "type": "string",
                                    "description": "String to replace with"
                                }
                            },
                            "required": ["filepath", "old_str", "new_str"]
                        }
                    }
                ]
            }
        ]

        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        GenerativeModel = getattr(genai, 'GenerativeModel')
        self.model = GenerativeModel(
            model_name,
            tools=tools,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        print("🤖 Using Google Gemini API")
    
    def init_anthropic(self):
        self.provider = "anthropic"
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        
        print("🤖 Using Anthropic Claude API")
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name == "read_file":
            return self.tools.read_file(args.get("filepath", ""))
        elif tool_name == "list_files":
            return self.tools.list_files(args.get("directory", "."))
        elif tool_name == "create_file":
            return self.tools.create_file(
                args.get("filepath", ""),
                args.get("content", "")
            )
        elif tool_name == "edit_file":
            return self.tools.edit_file(
                args.get("filepath", ""),
                args.get("old_str", ""),
                args.get("new_str", "")
            )
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def process_gemini_response(self, response) -> Tuple[str, bool]:
        text_parts = []
        has_tool_calls = False

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                elif hasattr(part, 'function_call'):
                    has_tool_calls = True

        return '\n'.join(text_parts), has_tool_calls
    
    def process_anthropic_response(self, message: str) -> str:
        if not self.client:
            return "Error: Anthropic client not initialized"

        client = self.client

        tools: List[ToolParam] = cast(List[ToolParam], [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to read"
                        }
                    },
                    "required": ["filepath"]
                }
            },
            {
                "name": "list_files",
                "description": "List files and directories in a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path (default: current directory)"
                        }
                    }
                }
            },
            {
                "name": "create_file",
                "description": "Create a new file with content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path for the new file"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["filepath", "content"]
                }
            },
            {
                "name": "edit_file",
                "description": "Edit a file by replacing exact string matches",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to edit"
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Exact string to replace"
                        },
                        "new_str": {
                            "type": "string",
                            "description": "String to replace with"
                        }
                    },
                    "required": ["filepath", "old_str", "new_str"]
                }
            }
        ])

        messages: List[MessageParam] = cast(List[MessageParam], [{"role": "user", "content": message}])
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1

            response = client.messages.create(
                model=str(self.model),
                max_tokens=4096,
                tools=tools,
                messages=messages
            )

            messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))

            final_text = []
            tool_uses = []
            
            for block in response.content:
                if block.type == "text":
                    final_text.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                return '\n'.join(final_text)

            tool_results = []
            for tool_use in tool_uses:
                print(f"\n🔧 Executing: {tool_use.name}")
                result = self.execute_tool(tool_use.name, tool_use.input)
                
                if "success" in result and result["success"]:
                    print(f"✅ {result.get('message', 'Success')}")
                elif "error" in result:
                    print(f"❌ {result['error']}")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result)
                })

            messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

        return "Max iterations reached"
    
    def chat(self, message: str) -> str:
        if self.provider == "gemini":
            if not self.model:
                return "Error: Gemini model not initialized"

            chat = self.model.start_chat()
            response = chat.send_message(message)
            
            iterations = 0
            while iterations < self.max_iterations:
                iterations += 1
                text, has_tools = self.process_gemini_response(response)
                
                if not has_tools:
                    return text

                if hasattr(response, 'candidates') and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call'):
                            fc = part.function_call
                            print(f"\n🔧 Executing: {fc.name}")
                            result = self.execute_tool(fc.name, dict(fc.args))

                            if "success" in result and result["success"]:
                                print(f"✅ {result.get('message', 'Success')}")
                            elif "error" in result:
                                print(f"❌ {result['error']}")

                            protos = getattr(genai, 'protos')
                            Content = getattr(protos, 'Content')
                            Part = getattr(protos, 'Part')
                            FunctionResponse = getattr(protos, 'FunctionResponse')

                            response = chat.send_message(
                                Content(
                                    parts=[
                                        Part(
                                            function_response=FunctionResponse(
                                                name=fc.name,
                                                response={"result": result}
                                            )
                                        )
                                    ]
                                )
                            )
            
            return "Max iterations reached"
        
        elif self.provider == "anthropic":
            return self.process_anthropic_response(message)

        return "Error: Unknown provider"
    
    def run(self):
        print("\n💬 Chat with your files")
        print("I can read, list, edit, and create files for you")
        print("Just ask naturally - type 'exit' when done\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye! 👋")
                    break
                
                if not user_input:
                    continue

                response = self.chat(user_input)
                
                if response:
                    print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def main():
    try:
        agent = AIAgent()
        agent.run()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
