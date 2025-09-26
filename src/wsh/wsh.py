"""Interactive shell for wikipedia-mcp server tools."""

import asyncio
from typing import Any, Dict, List
import cmd
import json

from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from wikisearch_agent.util import fetch_api_keys, setup_logging
from asyncio import AbstractEventLoop

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
        self.w_mcp_sever_params = StdioServerParameters(
            command=str(Path(__file__).parent.parent.parent / '.venv/bin/wikipedia-mcp'),
            args=['--enable-cache'],
            cwd='/tmp/'
        )
        self.loop: AbstractEventLoop | None = None
        self.w_session = ClientSession | None  # Wikipedia MCP server session.

    async def do_session(self):
        """Initialize the MCP session and discover available tools."""
        try:
            async with stdio_client(self.w_mcp_sever_params) as (w_read, w_write):
                async with ClientSession(read_stream=w_read, write_stream=w_write) as w_session:
                    self.w_session = w_session
                    await self.w_session.initialize()
                    tools = await load_mcp_tools(self.w_session)
                    self.tools = {
                        tool.name: tool for tool in tools
                    }
                    self.loop = asyncio.get_running_loop()

                    # Print available commands
                    console.print('\nAvailable commands:', style='bold green')
                    for name, tool in self.tools.items():
                        console.print(f'  {name}: {tool.description}')
                    console.print()
                    # Start the REPL
                    self.cmdloop()
                    console.print('Command loop terminated')
                console.print('Left ClientSession block')
            console.print('Left stdio_client block')
        except Exception as e:
            console.print(f'[red]Error during session initialization or execution:[/red]')
            console.print_exception()
            raise
        console.print('Finshed do_session')

    def do_quit(self, _: str) -> bool:
        """Exit the shell."""
        return True

    def do_EOF(self, _: str) -> bool:
        """Exit the shell."""
        return True

    def do_ls(self, _: str):
        files = [str(f) for f in Path('.').glob('*')]
        self.columnize(files)

    def default(self, line: str) -> None:
        """Handle tool execution for unknown commands."""
        cmd_parts = line.split(maxsplit=1)
        if not cmd_parts:
            return

        tool_name = cmd_parts[0].lower()
        if tool_name not in self.tools:
            console.print(f'[red]Unknown command: {tool_name}[/red]')
            return

        # Parse arguments as JSON
        args_str = cmd_parts[1] if len(cmd_parts) > 1 else ''
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            console.print('[red]Error: Arguments must be valid JSON[/red]')
            return

        # Execute the tool
        if self.loop is None:
            console.print('[red]Error![/red] Event loop not running. This is a core bug.')
        else:
            self._execute_tool(tool_name=tool_name, args=args)
            # future = self.loop.create_task(self._execute_tool(tool_name, args))
            # self.loop.run_until_complete(future=future)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Execute a tool and display its results."""
        if self.w_session is None:
            console.print('[red]Error: Session not initialized[/red]')
            return

        if tool_name not in self.tools:
            console.print(f'[red]Unknown tool: {tool_name}[/red]')
            return

        try:
            tool = self.tools[tool_name]
            console.print(f'Entering {tool_name} => {tool.description}')
            result = tool.ainvoke(args)
            console.print(f'Completed tool {tool_name}')

            # Pretty print the result
            if isinstance(result, (dict, list)):
                json_str = json.dumps(result, indent=2)
                syntax = Syntax(json_str, 'json', theme='monokai')
                console.print(syntax)
            else:
                console.print(str(result))

        except Exception as e:
            console.print(f'[red]Error executing {tool_name}[/red]')
            console.print_exception()

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
                if tool.args is not None:
                    console.print('Arguments:')
                    syntax = Syntax(
                        json.dumps(tool.args, indent=2),
                        'json',
                        theme='monokai'
                    )
                    console.print(syntax)
                    console.print('===')
                    console.print_json(json.dumps(tool.args), indent=2)
            else:
                console.print(f'[red]No help available for: {arg}[/red]')

def main():
    """Main entry point for the Wikipedia shell."""
    shell = WikipediaShell()
    try:
        # Initialize the session
        asyncio.run(shell.do_session())
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
