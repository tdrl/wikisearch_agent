"""Interactive shell for wikipedia-mcp server tools."""

import asyncio
from typing import Any, Dict, List
import cmd
import json
import subprocess
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from wikisearch_agent.util import fetch_api_keys, setup_logging

console = Console()

class WikipediaShell(cmd.Cmd):
    """Interactive shell for wikipedia-mcp tools."""

    intro = 'Welcome to the Wikipedia MCP shell. Type help or ? to list commands.\n'
    prompt = 'wsh> '

    def __init__(self):
        super().__init__()
        self.logger = setup_logging(loglevel='INFO')
        self.tools: Dict[str, Any] = {}
        self.client: Any = None
        self.process: subprocess.Popen | None = None

    async def init_session(self):
        """Initialize the MCP session and discover available tools."""
        try:
            params = StdioServerParameters(command='python -m wikipedia_mcp.server')
            # Create and initialize the MCP client
            async with stdio_client(params) as client:
                self.client = client

            # Load the tools
            tools = await load_mcp_tools(self.client)
            self.tools = {
                tool.name: tool for tool in tools
            }

            # Print available commands
            console.print('\nAvailable commands:', style='bold green')
            for name, tool in self.tools.items():
                console.print(f'  {name}: {tool.description}')
            console.print()

        except Exception as e:
            console.print(f'[red]Error initializing session: {str(e)}[/red]')
            if self.process:
                self.process.terminate()
            raise

    def do_quit(self, _: str) -> bool:
        """Exit the shell."""
        return True

    def default(self, line: str) -> None:
        """Handle tool execution for unknown commands."""
        cmd_parts = line.split()
        if not cmd_parts:
            return

        tool_name = cmd_parts[0].lower()
        if tool_name not in self.tools:
            console.print(f'[red]Unknown command: {tool_name}[/red]')
            return

        # Parse arguments as JSON
        args_str = ' '.join(cmd_parts[1:])
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            console.print('[red]Error: Arguments must be valid JSON[/red]')
            return

        # Execute the tool
        asyncio.run(self._execute_tool(tool_name, args))

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Execute a tool and display its results."""
        if not self.client:
            console.print('[red]Error: Session not initialized[/red]')
            return

        if tool_name not in self.tools:
            console.print(f'[red]Unknown tool: {tool_name}[/red]')
            return

        try:
            tool = self.tools[tool_name]
            result = await tool.invoke(args)

            # Pretty print the result
            if isinstance(result, (dict, list)):
                json_str = json.dumps(result, indent=2)
                syntax = Syntax(json_str, 'json', theme='monokai')
                console.print(syntax)
            else:
                console.print(str(result))

        except Exception as e:
            console.print(f'[red]Error executing {tool_name}: {str(e)}[/red]')

    def do_help(self, arg: str) -> None:
        """List available commands or get help for a specific command."""
        if not arg:
            console.print('\nAvailable commands:', style='bold green')
            console.print('  quit: Exit the shell')
            console.print('  help: Show this help message')
            for name, tool in self.tools.items():
                console.print(f'  {name}: {tool.description}')
        else:
            if arg.lower() in self.tools:
                tool = self.tools[arg.lower()]
                console.print(f'\n{arg}:', style='bold green')
                console.print(f'Description: {tool.description}')
                if tool.parameters:
                    console.print('Parameters:')
                    syntax = Syntax(
                        json.dumps(tool.parameters, indent=2),
                        'json',
                        theme='monokai'
                    )
                    console.print(syntax)
            else:
                console.print(f'[red]No help available for: {arg}[/red]')

def main():
    """Main entry point for the Wikipedia shell."""
    shell = WikipediaShell()
    try:
        # Initialize the session
        asyncio.run(shell.init_session())
        # Start the REPL
        shell.cmdloop()
    except KeyboardInterrupt:
        console.print('\nGoodbye!')
    except Exception as e:
        console.print(f'[red]Error: {str(e)}[/red]')
    finally:
        if shell.client:
            # The client's context manager will handle cleanup
            pass

if __name__ == '__main__':
    main()
