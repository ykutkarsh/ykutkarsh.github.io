import pytest

from aab.memory.database import ExperienceStore
from aab.research.search import PublicSearch, SearchResult
from aab.research.source_evaluator import SourceEvaluator
from aab.research.web_reader import WebReader


def test_search_uses_injected_public_transport_and_validates_inputs():
    search = PublicSearch(lambda query, limit: [SearchResult("Result", "https://example.test", query)])
    assert search.search("climate", 1)[0].title == "Result"
    with pytest.raises(ValueError):
        search.search("", 1)
    with pytest.raises(ValueError):
        search.search("x", 0)


def test_search_requires_configuration_without_transport(monkeypatch):
    monkeypatch.delenv("AAB_SEARCH_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AAB_SEARCH_ENDPOINT"):
        PublicSearch().search("anything")


def test_web_reader_extracts_text_and_ignores_script_content():
    html = b"<html><title>Example</title><script>ignore()</script><p>Useful evidence.</p></html>"
    reader = WebReader(lambda url, max_bytes: (html, "text/html"))
    document = reader.read("https://example.test/page")
    assert document.title == "Example"
    assert document.text == "Example Useful evidence."
    with pytest.raises(ValueError):
        reader.read("file:///tmp/private")


def test_source_quality_and_verified_discovery_persist_across_reopen(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        evaluator = SourceEvaluator(store)
        documents = [
            WebReader(lambda url, limit: (b"<title>A</title><p>Water boils at 100 degrees Celsius at sea level.</p>", "text/html")).read("https://example.edu/a"),
            WebReader(lambda url, limit: (b"<title>B</title><p>Water boils at 100 degrees Celsius at sea level.</p>", "text/html")).read("https://example.gov/b"),
        ]
        discovery = evaluator.store_discovery("Water boils at 100 degrees Celsius at sea level", documents)
        assert discovery.status == "verified"
        assert discovery.confidence > 0.5
    with ExperienceStore(tmp_path / "memory.db") as reopened:
        row = reopened._connection.execute("SELECT status, citations_json FROM discoveries").fetchone()
        assert row["status"] == "verified"
        assert "example.edu" in row["citations_json"]


def test_conflicting_evidence_is_labeled(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        evaluator = SourceEvaluator(store)
        docs = [
            WebReader(lambda url, limit: (b"<p>Claim alpha is true and supported.</p>", "text/html")).read("https://one.test"),
            WebReader(lambda url, limit: (b"<p>Claim beta is true and opposed.</p>", "text/html")).read("https://two.test"),
        ]
        discovery = evaluator.store_discovery("Claim alpha is true", docs)
        assert discovery.status == "conflicted"
