"""
LLM Analiz Testleri — Gemini / Ollama entegrasyonu, JSON parse, hata yönetimi.
Gerçek API çağrısı yapılmaz; mock kullanılır.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from llm_analyzer import analyze_stock, _parse_response


class TestParseResponse:
    def test_valid_json(self, sample_report):
        raw = json.dumps(sample_report, ensure_ascii=False)
        result = _parse_response(raw)
        assert result == sample_report

    def test_json_with_markdown_fences(self, sample_report):
        raw = f"```json\n{json.dumps(sample_report)}\n```"
        result = _parse_response(raw)
        assert "analiz_ozeti" in result

    def test_invalid_json_returns_fallback(self):
        result = _parse_response("Bu JSON değil!!!")
        assert "analiz_ozeti" in result
        assert result["risk_analizi"]["risk_skoru"] == 50

    def test_empty_string_returns_fallback(self):
        result = _parse_response("")
        assert isinstance(result, dict)
        assert "risk_analizi" in result

    def test_valid_report_has_required_fields(self, sample_report):
        raw = json.dumps(sample_report)
        result = _parse_response(raw)
        assert "genel_gorus" in result["analiz_ozeti"]
        assert "risk_skoru" in result["risk_analizi"]


class TestAnalyzeStock:
    @patch("llm_analyzer.is_ollama_provider", return_value=False)
    @patch("llm_analyzer.is_llm_configured", return_value=True)
    @patch("llm_analyzer.genai.Client")
    def test_successful_analysis(
        self,
        mock_client_cls,
        _mock_llm_ok,
        _mock_ollama,
        sample_stock_data,
        sample_report,
        mock_gemini_response,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_gemini_response

        with patch("llm_analyzer.GEMINI_API_KEY", "AIzaSyB12345678901234567890123456789012"):
            result = analyze_stock(sample_stock_data, {}, "Test haberler")

        assert "analiz_ozeti" in result
        assert result["risk_analizi"]["risk_skoru"] == 42

    @patch("llm_analyzer.is_ollama_provider", return_value=False)
    @patch("llm_analyzer.is_llm_configured", return_value=True)
    @patch("llm_analyzer.genai.Client")
    def test_rate_limit_retry(self, mock_client_cls, _mock_llm_ok, _mock_ollama, sample_stock_data, sample_report):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        good_resp = MagicMock()
        good_resp.text = json.dumps(sample_report)

        mock_client.models.generate_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            good_resp,
        ]

        with (
            patch("llm_analyzer.GEMINI_API_KEY", "AIzaSyB12345678901234567890123456789012"),
            patch("llm_analyzer.time.sleep"),
        ):
            result = analyze_stock(sample_stock_data, {}, "")

        assert "analiz_ozeti" in result
        assert mock_client.models.generate_content.call_count == 2

    @patch("llm_analyzer.is_ollama_provider", return_value=False)
    @patch("llm_analyzer.is_llm_configured", return_value=True)
    @patch("llm_analyzer.genai.Client")
    def test_auth_error_raises_immediately(self, mock_client_cls, _mock_llm_ok, _mock_ollama, sample_stock_data):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("403 PERMISSION_DENIED")

        with (
            patch("llm_analyzer.GEMINI_API_KEY", "AIzaSyB12345678901234567890123456789012"),
            pytest.raises(RuntimeError, match="Gemini API yetki hatası"),
        ):
            analyze_stock(sample_stock_data, {}, "")

        assert mock_client.models.generate_content.call_count == 1

    @patch("llm_analyzer.is_ollama_provider", return_value=False)
    @patch("llm_analyzer.is_llm_configured", return_value=False)
    def test_missing_api_key_raises(self, _mock_ollama, _mock_llm, sample_stock_data):
        with (
            patch("llm_analyzer.GEMINI_API_KEY", ""),
            pytest.raises(RuntimeError, match="GEMINI_API_KEY"),
        ):
            analyze_stock(sample_stock_data, {}, "")

    @patch("llm_analyzer.is_ollama_provider", return_value=True)
    @patch("llm_analyzer.is_llm_configured", return_value=True)
    @patch("llm_analyzer.requests.post")
    def test_ollama_success(
        self, mock_post, _mock_llm, _mock_ollama, sample_stock_data, sample_report
    ):
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={
                    "message": {"content": json.dumps(sample_report, ensure_ascii=False)},
                }
            ),
        )
        result = analyze_stock(sample_stock_data, {}, "")
        assert result["risk_analizi"]["risk_skoru"] == 42
        mock_post.assert_called_once()


class TestReportStructure:
    @patch("llm_analyzer.is_ollama_provider", return_value=False)
    @patch("llm_analyzer.is_llm_configured", return_value=True)
    @patch("llm_analyzer.genai.Client")
    def test_report_risk_score_is_integer(
        self, mock_client_cls, _mock_llm_ok, _mock_ollama, sample_stock_data, sample_report, mock_gemini_response
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_gemini_response

        with patch("llm_analyzer.GEMINI_API_KEY", "AIzaSyB12345678901234567890123456789012"):
            result = analyze_stock(sample_stock_data, {}, "")

        risk = result["risk_analizi"]["risk_skoru"]
        assert isinstance(risk, int)
        assert 0 <= risk <= 100
