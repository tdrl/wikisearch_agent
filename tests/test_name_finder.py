from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel
import pytest
from wikisearch_agent.name_finder import App, NameFinderAppState
from wikisearch_agent.schemas.person import PersonInfo, ArticleNames
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, RunnableConfig
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel, FakeChatModel, FakeListChatModel
from langgraph.graph import END, START, StateGraph
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic_core import to_json


@pytest.fixture
def alice_in_wonderland() -> PersonInfo:
    return PersonInfo(
        birth_name='Alice Liddel',
        best_known_as='Alice',
        alternate_names=['Alice in Wonderland'],
        best_known_for='Falling down the rabbit hole',
        is_human=True,
        is_real=False,
        birth_year=1887,
        birth_day=None,
        birth_month=None,
        assigned_gender_at_birth='Female',
        gender_identity='Female',
        continent_of_origin='Europe',
        country_of_origin='England',
        locality_of_origin='Oxford'
    )


@pytest.fixture
def alice_in_wonderland_str(alice_in_wonderland) -> str:
    return str(to_json(alice_in_wonderland), encoding='utf8')


class StructuredFakeListChatModel(FakeListChatModel):
    def with_structured_output(self, schema: Dict | type, *, include_raw: bool = False, **kwargs: Any):
        return self

    def invoke(self, input, config: RunnableConfig | None = None, *, stop: list[str] | None = None, **kwargs: Any) -> BaseMessage:
        raw = super().invoke(input, config, stop=stop, **kwargs)
        return JsonOutputParser().parse(raw.content)  # type: ignore


class TestNameFinder:

    def test_simple_prompt_interpolation(self):
        t1 = ChatPromptTemplate([
            ('system', 'you are a helpful bot'),
            ('user', 'What is the answer to {question}?')
        ])
        result = t1.invoke({'unused': 'stuff', 'question': 'the ultimate question'})
        print(result)
        msgs = result.to_messages()
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == 'you are a helpful bot'
        assert isinstance(msgs[1], HumanMessage)
        assert msgs[1].content == 'What is the answer to the ultimate question?'

    def test_simple_chat_flow_with_bot(self):
        t1 = ChatPromptTemplate([
            ('system', 'you are a helpful bot'),
            ('user', 'what is your name?'),
            ('user', 'placeholder: {placeholder}'),
        ])
        fake_llm = FakeListChatModel(name='fakPG-4o-pico', responses=['alpha', 'beta', 'gamma', 'delta'])
        result = (t1 | fake_llm).invoke(input={'placeholder': 'Grand Canyon'})
        assert isinstance(result, AIMessage)
        assert result.content == 'alpha'

    def test_multistep_bot_chat_flow(self):
        t1 = ChatPromptTemplate([
            ('system', 'you are a helpful bot'),
            ('user', 'what is your name?'),
            ('user', 'placeholder: {placeholder}'),
        ])
        fake_llm = FakeListChatModel(name='fakPG-4o-pico', responses=['alpha', 'beta', 'gamma', 'delta'])
        t2 = ChatPromptTemplate([
            ('user', "that's so FAaaascinating!")
        ])
        result = (t1 | fake_llm | RunnableLambda(lambda x: {'messages': [x]}) | t2 | fake_llm).invoke(input={'placeholder': 'Grand Canyon'})
        assert isinstance(result, AIMessage)
        assert result.content == 'beta'
