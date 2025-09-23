"""Interactive shell for wikipedia-mcp server tools."""

import asyncio
from typing import Any, Dict, List
import cmd
import json
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from mcp import ClientSession, StdioServerParameters
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
        self.session: ClientSession | None = None

    async def init_session(self):
        """Initialize the MCP session and discover available tools."""
        try:
            # Set up the MCP client session
            server_params = StdioServerParameters([
                'python', '-m', 'wikipedia_mcp.server'
            ])

            # Start the MCP session
            self.session = await stdio_client(server_params)

            # Discover available tools
            tools_response = await self.session.discover_tools()
            self.tools = {
                tool.name: tool for tool in tools_response.tools
            }

            # Print available commands
            console.print('\nAvailable commands:', style='bold green')
            for name, tool in self.tools.items():
                console.print(f'  {name}: {tool.description}')
            console.print()

        except Exception as e:
            console.print(f'[red]Error initializing session: {str(e)}[/red]')
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
        if not self.session:
            console.print('[red]Error: Session not initialized[/red]')
            return

        try:
            result = await self.session.invoke_tool(tool_name, args)

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
    finally:
        if shell.session is not None:
            asyncio.run(shell.session.aclose())

if __name__ == '__main__':
    main()
