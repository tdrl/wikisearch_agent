"""Core wikisearch agent application infra."""

from typing import Optional
import asyncio
from wikisearch_agent.util import fetch_api_keys, setup_logging, prompt_template_from_file
from dataclasses import asdict
import langsmith
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.runnables.base import Runnable
from langchain_core.tools.base import BaseTool
from langchain_core.output_parsers import JsonOutputParser, JsonOutputToolsParser, StrOutputParser
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from pathlib import Path
import json


class PersonInfo(BaseModel):
      """Information about a single person-entity."""
      birth_name: str = Field(description='Full birth name')
      best_known_as: str = Field(description='Name by which this person is most widely known')
      alternate_names: list[str] = Field(description='List of other or alternate names, such as aliases, '
                                                     'pen names, noms de guerre, nicknames, etc.')
      best_known_for: str = Field(description='A one sentence description of why this person is '
                                              'noteworthy what they are known for')
      is_real: bool = Field(description='True iff this entity is real (as opposed to fictional, imaginary,'
                                        ' a character in a book or movie, etc.)')
      is_human: bool = Field(description='True iff this entity is a human (as opposed to, say, an animal, '
                                         'pet, alien, monster, imaginary being, etc.)')
      birth_year: Optional[int] = Field(description='Year of birth (if known). Use a negative value for BCE '
                                                    'dates; positive for CE dates.')
      birth_month: Optional[int] = Field(description='Month of birth (if known), starting with January = 1 through December = 12.')
      birth_day: Optional[int] = Field(description='Day of birth, within month, in the range [1, 31].')
      assigned_gender_at_birth: str = Field(description='Entitie''s gender, as assigned at birth. '
                                                        'Possible values: Male|Female|Nonbinary|Two spirit|Other|Unknown')
      gender_identity: str = Field(description='Entitie''s self-identified gender identity. May or '
                                               'may not be the same as their assigned gender at birth. Possible values: '
                                               'Male|Female|Nonbinary|Two spirit|Other|Unknown')
      continent_of_origin: str = Field(description='The continent or island on which they were born. '
                                                   'Possible values: Africa|Asia|Australia|Europe|South America|North '
                                                   'America|Antarctica|Island name')
      country_of_origin: str = Field(description='The country in which they were born (if any), as that '
                                                 'country was known at the time of their birth. Country '
                                                 'name, or null')
      locality_of_origin: str = Field(description='City or town or village name, or null')


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

    def build_entity_analyzer_agent(self, tools: list[BaseTool]) -> tuple[Runnable, JsonOutputParser]:
        name_data_parser = JsonOutputParser(pydantic_object=PersonInfo)
        name_agent_prompt_templ = prompt_template_from_file(Path(__file__).parent.parent.parent / 'prompts/name_extractor_agent.yaml')
        agent = create_react_agent(model=self.llm, tools=tools)
        chain = name_agent_prompt_templ | agent
        return chain, name_data_parser

    async def run(self):
        try:
            with langsmith.tracing_context(project_name='hrl/wikisearch', enabled=True, client=self.tracing_client):
                self.logger.info('Starting Wikisearch agent')
                async with stdio_client(self.w_mcp_sever_params) as (w_read, w_write):
                    async with ClientSession(read_stream=w_read, write_stream=w_write) as w_session:
                        self.logger.debug('Initilializing MCP server sessions')
                        await w_session.initialize()
                        self.logger.debug('MCP session initialized')
                        wikipedia_tools = await load_mcp_tools(w_session)
                        self.logger.debug('Got tools', tools=wikipedia_tools)
                        entity_research_agent, name_data_parser = self.build_entity_analyzer_agent(tools=wikipedia_tools)
                        context = {
                            # 'person': 'Mark Twain',
                            'person': 'Jane Doe',
                            'format_instructions': name_data_parser.get_format_instructions(),
                        }
                        response = await entity_research_agent.ainvoke(context)
                        self.logger.info('Got chain response', type=type(response), keys=list(response.keys()))
                        structured_response = name_data_parser.parse(response['messages'][-1].content)
                        self.logger.info('Agent final state', response=response['messages'][-1].content)
                        self.logger.info('JSON result', structured_response=structured_response)
                        Path('/tmp/heather/agent_out.json').write_text(json.dumps(structured_response, indent=2))
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise

async def main():
    app = App()
    await app.run()

if __name__ == '__main__':
    asyncio.run(main=main())