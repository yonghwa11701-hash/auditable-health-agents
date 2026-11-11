import os
import json
import pytest
from unittest import mock
from agent.evaluator import evaluate_run, BinaryEvalResponse

@pytest.fixture
def mock_genai_client():
    with mock.patch("agent.evaluator.make_client") as mk:
        client = mock.MagicMock()
        mk.return_value = client
        yield client

def test_evaluate_run(tmp_path, mock_genai_client):
    run_log = tmp_path / "run.jsonl"
    run_log.write_text(
        json.dumps({"type": "orchestrator", "text": "Top pair: erlotinib + gefitinib"}) + "\n" +
        json.dumps({"type": "gates", "results": [{"gate": "citations_grounded", "pass": True}]}) + "\n"
    )
    
    mock_response = mock.MagicMock()
    mock_response.text = json.dumps({
        "persona": "Biostatistician",
        "scorecard": {"Are all generated claims supported...": True},
        "rationale": "Grounded citations gate passed.",
        "composite_pass": True
    })
    
    mock_genai_client.models.generate_content.return_value = mock_response
    
    results = evaluate_run(str(run_log))
    assert len(results) == 6  # 6 personas
    assert results[0].composite_pass is True
    assert results[0].rationale == "Grounded citations gate passed."

