"""Core wikisearch agent application infra."""

from typing import Optional
from wikisearch_agent.util import fetch_api_keys, setup_logging
from dataclasses import asdict
import langsmith
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class App:
    """The main application harness.

    Fetches secrets, configures the LLM, configures tracing, and provides an entrypoint for
    running the chain.
    """
    def __init__(self, argv: Optional[list[str]] = None) -> None:
        # TODO(heather) Handle command-line, if necessary. For the moment, assume
        # logging defaults are good.
        self.logger = setup_logging()
        self.secrets = fetch_api_keys()
        if self.secrets.openai_api is None:
            raise PermissionError("Don't have access to OpenAI API")
        self.logger.debug('Fetch keys', keys=[k for k in asdict(self.secrets)])
        self.tracing_client = langsmith.Client(api_key=self.secrets.langsmith_api)
        # TODO(heather) Consider factoring out the model name and temperature to command line args.
        self.llm = ChatOpenAI(model='gpt-5-mini', api_key=self.secrets.openai_api) # type: ignore

    def run(self):
        try:
            with langsmith.tracing_context(project_name='hrl/wikisearch', enabled=True, client=self.tracing_client):
                self.logger.info('Starting Wikisearch agent')
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ('system', 'You are a helpful research librarian with a sense of humor.'),
                        ('user', 'What is the answer to the ultimate question?')
                    ]
                )
                chain = prompt | self.llm
                response = chain.invoke({})
                self.logger.info('OpenAI query', response=response.text())
        except Exception as e:
            self.logger.exception('Uncaught error somewhere in the code (hopeless).', exc_info=e)
            raise


if __name__ == '__main__':
    App().run()
