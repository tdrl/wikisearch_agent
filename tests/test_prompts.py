import pytest
from pathlib import Path
import yaml
from langchain_core.prompts import ChatPromptTemplate
from wikisearch_agent.util import prompt_template_from_file

@pytest.fixture
def project_root_path(request: pytest.FixtureRequest) -> Path:
    return request.config.rootpath


class TestPrompts:

    def test_load_name_extractor_agent_prompt(self, project_root_path: Path):
        prompt_file = project_root_path / 'prompts/name_extractor_agent.yaml'
        assert prompt_file.exists()
        templ = prompt_template_from_file(prompt_file)
        assert len(templ.messages) == 2
        assert 'person' in templ.input_variables