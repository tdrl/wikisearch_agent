"""Core wikisearch agent application infra."""

from typing import Optional
import asyncio
from wikisearch_agent.util import fetch_api_keys, setup_logging, prompt_template_from_file
from dataclasses import asdict
import langsmith
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import JsonOutputParser, JsonOutputToolsParser, StrOutputParser
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from pathlib import Path


class App:
    """The main application harness.

    Fetches secrets, configures the LLM, configures tracing, and provides an entrypoint for
    running the chain.
    """
    def __init__(self, argv: Optional[list[str]] = None) -> None:
        # TODO(heather) Handle command-line, if necessary. For the moment, assume
        # logging defaults are good.
        self.logger = setup_logging(loglevel='INFO')
        self.secrets = fetch_api_keys()
        if self.secrets.openai_api is None:
            raise PermissionError("Don't have access to OpenAI API")
        self.logger.debug('Fetch keys', keys=[k for k in asdict(self.secrets)])
        self.tracing_client = langsmith.Client(api_key=self.secrets.langsmith_api)
        # TODO(heather) Consider factoring out the model name and temperature to command line args.
        self.llm = ChatOpenAI(model='gpt-5-mini', api_key=self.secrets.openai_api) # type: ignore
        self.w_mcp_sever_params = StdioServerParameters(
            command=str(Path(__file__).parent.parent.parent / '.venv/bin/wikipedia-mcp'),
            args=[],
            cwd='/tmp/'
        )

    async def run(self):
        try:
            with langsmith.tracing_context(project_name='hrl/wikisearch', enabled=True, client=self.tracing_client):
                self.logger.info('Starting Wikisearch agent')
                async with stdio_client(self.w_mcp_sever_params) as (w_read, w_write):
                    async with ClientSession(read_stream=w_read, write_stream=w_write) as w_session:
                        self.logger.debug('Initilializing MCP server sessions')
                        await w_session.initialize()
                        self.logger.debug('MCP session initialized')
                        tools = await load_mcp_tools(w_session)
                        self.logger.debug('Got tools', tools=tools)
                        context = {
                            'person': 'Mark Twain'
                        }
                        name_agent_prompt_templ = prompt_template_from_file(Path(__file__).parent.parent.parent / 'prompts/name_extractor_agent.yaml')
                        # chain = prompt | self.llm.bind_tools(tools=tools) | JsonOutputParser()
                        # response = await chain.ainvoke(context)
                        agent = create_react_agent(model=self.llm, tools=tools)
                        chain = name_agent_prompt_templ | agent
                        response = await chain.ainvoke(context)
                        self.logger.info('OpenAI query', response=response['messages'][-1].content)
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise

async def main():
    app = App()
    await app.run()

if __name__ == '__main__':
    asyncio.run(main=main())