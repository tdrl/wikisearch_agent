"""Core wikisearch agent application infra."""

from typing import Optional
import asyncio
from wikisearch_agent.util import (
    fetch_api_keys,
    setup_logging,
    prompt_template_from_file,
    get_default_working_dir
)
from wikisearch_agent.schemas.person import PersonInfo, ArticleNames
from dataclasses import asdict
import langsmith
from langchain_openai import ChatOpenAI
from pydantic_core import to_json
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_core.tools.base import BaseTool
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from pathlib import Path
import json


class NameFinderAppState(AgentState):
    """State for the NameFinder app: Detailed info about a single entity, and all co-occurring names."""
    target_person: str | None
    entity_data: PersonInfo | None
    article_name_data: ArticleNames | None


class App:
    """The main application harness.

    Fetches secrets, configures the LLM, configures tracing, and provides an entrypoint for
    running the chain.
    """
    def __init__(self, argv: Optional[list[str]] = None) -> None:
        # TODO(heather) Handle command-line, if necessary. For the moment, assume
        # logging defaults are good.
        self.logger = setup_logging(loglevel='INFO')
        # TODO(heather) CLI flag to specify a real output location.
        self.work_dir = get_default_working_dir()
        self.prompts_dir = Path(__file__).parent.parent.parent / 'prompts'
        self.secrets = fetch_api_keys()
        if self.secrets.openai_api is None:
            raise PermissionError("Don't have access to OpenAI API")
        self.logger.debug('Fetch keys', keys=[k for k in asdict(self.secrets)])
        # TODO(heather) Consider factoring out the model name and temperature to command line args.
        self.reseacher = ChatOpenAI(model='gpt-5-mini', api_key=self.secrets.openai_api) # type: ignore
        self.scraper = ChatOpenAI(model='gpt-5-nano', api_key=self.secrets.openai_api).with_structured_output(schema=ArticleNames)  # type: ignore
        self.wikipedia_tools: list[BaseTool] = []
        self.reseacher_prompt = prompt_template_from_file(self.prompts_dir / 'name_extractor_agent.yaml')
        self.name_locator_prompt = prompt_template_from_file(self.prompts_dir / 'name_scraper_prompt.yaml')
        self.tracing_client = langsmith.Client(api_key=self.secrets.langsmith_api)
        self.w_mcp_sever_params = StdioServerParameters(
            command=str(Path(__file__).parent.parent.parent / '.venv/bin/wikipedia-mcp'),
            args=['--enable-cache'],
            cwd='/tmp/'
        )

    async def build_entity_analyzer_node(self, state: NameFinderAppState) -> NameFinderAppState:
        agent = create_react_agent(model=self.reseacher,
                                   tools=self.wikipedia_tools,
                                   response_format=PersonInfo,
                                   name='PersonResearcher')
        chain = self.reseacher_prompt | agent
        result = await chain.ainvoke({
            'person': state['target_person'],
            # TODO(heather): There has to be a better way to handle this.
            'format_instructions': JSON_FORMAT_INSTRUCTIONS.format(schema=json.dumps(PersonInfo.model_json_schema())),
        })
        update = NameFinderAppState(entity_data=result['structured_response'],
                                    messages=result['messages'])  # type: ignore
        return update

    async def build_name_locator_node(self, state: NameFinderAppState) -> NameFinderAppState:
        docs = [x.content for x in state['messages'] if isinstance(x, ToolMessage)]
        docs = '\n'.join(docs)  # type: ignore
        result = await (self.name_locator_prompt | self.scraper).ainvoke({'all_docs': docs})
        update = NameFinderAppState(article_name_data=result)  # type: ignore
        return update

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(state_schema=NameFinderAppState)
        ENTITY_RESEARCHER_NODE = 'Entity Researcher'
        NAME_FINDER_NODE = 'Names Finder'
        builder.add_node(ENTITY_RESEARCHER_NODE, self.build_entity_analyzer_node)
        builder.add_edge(START, ENTITY_RESEARCHER_NODE)
        builder.add_node(NAME_FINDER_NODE, self.build_name_locator_node)
        builder.add_edge(ENTITY_RESEARCHER_NODE, NAME_FINDER_NODE)
        builder.add_edge(NAME_FINDER_NODE, END)
        return builder.compile()

    async def run(self):
        try:
            with langsmith.tracing_context(project_name='hrl/wikisearch', enabled=True, client=self.tracing_client):
                self.logger.info('Starting Wikisearch name finder agent')
                async with stdio_client(self.w_mcp_sever_params) as (w_read, w_write):
                    async with ClientSession(read_stream=w_read, write_stream=w_write) as w_session:
                        self.logger.debug('Initilializing MCP server sessions')
                        await w_session.initialize()
                        self.logger.debug('MCP session initialized')
                        self.wikipedia_tools = await load_mcp_tools(w_session)
                        self.logger.debug('Got tools', tools=self.wikipedia_tools)
                        graph = self.build_graph()
                        start_state = NameFinderAppState(messages=[],
                                                         remaining_steps=20,
                                                         target_person='Kitty Dukakis',
                                                         entity_data=None,
                                                         article_name_data=None)
                        result = await graph.ainvoke(start_state)
                        self.logger.info('Got chain response', type=type(result), keys=list(result.keys()))
                        self.logger.info('Agent final state', result=result['messages'][-1].content)
                        self.logger.info('JSON entity result', structured_response=result['entity_data'])
                        (self.work_dir / 'agent_out.json').write_bytes(to_json(result, indent=2))
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


async def amain():
    """Asynchronous program entry point."""
    await App().run()


def main():
    """Synchronous program entry point."""
    asyncio.run(main=amain())


if __name__ == '__main__':
    """Command-line direct invocation entry point."""
    main()