import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.recommendation_agent import RecommendationAgent
from ai.model_manager import ModelManager

@pytest.fixture
def model_manager():
    return ModelManager()


@pytest.fixture
def recommendation_agent():
    return RecommendationAgent()


def test_model_manager_initialization():
    manager = ModelManager()
    assert manager is not None


def test_recommendation_agent_initialization():
    agent = RecommendationAgent()
    assert agent is not None


def test_agent_has_required_methods():
    agent = RecommendationAgent()
    assert hasattr(agent, 'generate_recommendations')
    assert hasattr(agent, 'process_query')


@pytest.mark.asyncio
async def test_agent_basic_functionality():
    agent = RecommendationAgent()
    assert agent is not None
