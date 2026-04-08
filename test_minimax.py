"""Unit and integration tests for MiniMax provider support in Synesthesia."""
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Make sure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import LLMBridge


class TestLLMBridgeMinimaxDetection(unittest.TestCase):
    """Unit tests: model name routing."""

    def setUp(self):
        self.bridge = LLMBridge()

    def test_minimax_model_detected(self):
        self.assertTrue(self.bridge._is_minimax_model("MiniMax-M2.7"))

    def test_minimax_highspeed_detected(self):
        self.assertTrue(self.bridge._is_minimax_model("MiniMax-M2.7-highspeed"))

    def test_minimax_case_insensitive(self):
        self.assertTrue(self.bridge._is_minimax_model("minimax-m2.7"))

    def test_non_minimax_model(self):
        self.assertFalse(self.bridge._is_minimax_model("qwen3-vl-8b"))
        self.assertFalse(self.bridge._is_minimax_model("gpt-4"))
        self.assertFalse(self.bridge._is_minimax_model("llama-3"))


class TestLLMBridgeBaseURL(unittest.TestCase):
    """Unit tests: correct base URL selected per model."""

    def setUp(self):
        self.bridge = LLMBridge()

    def test_minimax_uses_minimax_base_url(self):
        url = self.bridge._get_base_url("MiniMax-M2.7")
        self.assertEqual(url, config.MINIMAX_BASE_URL)
        self.assertTrue(url.startswith("https://api.minimax.io"))

    def test_local_model_uses_lm_studio_url(self):
        url = self.bridge._get_base_url("qwen3-vl-8b")
        self.assertEqual(url, config.LM_STUDIO_URL)


class TestLLMBridgeHeaders(unittest.TestCase):
    """Unit tests: auth headers only added for MiniMax when API key is set."""

    def test_no_header_without_api_key(self):
        original = config.MINIMAX_API_KEY
        try:
            config.MINIMAX_API_KEY = ""
            bridge = LLMBridge()
            headers = bridge._get_headers("MiniMax-M2.7")
            self.assertNotIn("Authorization", headers)
        finally:
            config.MINIMAX_API_KEY = original

    def test_bearer_header_with_api_key(self):
        original = config.MINIMAX_API_KEY
        try:
            config.MINIMAX_API_KEY = "test-key-123"
            bridge = LLMBridge()
            headers = bridge._get_headers("MiniMax-M2.7")
            self.assertEqual(headers.get("Authorization"), "Bearer test-key-123")
        finally:
            config.MINIMAX_API_KEY = original

    def test_no_header_for_local_model(self):
        original = config.MINIMAX_API_KEY
        try:
            config.MINIMAX_API_KEY = "test-key-123"
            bridge = LLMBridge()
            headers = bridge._get_headers("qwen3-vl-8b")
            self.assertNotIn("Authorization", headers)
        finally:
            config.MINIMAX_API_KEY = original


class TestLLMBridgeTemperatureClamp(unittest.TestCase):
    """Unit tests: temperature clamped for MiniMax (must be in (0.0, 1.0])."""

    def test_temperature_clamped_above_zero(self):
        """MiniMax does not accept temperature=0; should be clamped to 0.01."""
        original_key = config.MINIMAX_API_KEY
        try:
            config.MINIMAX_API_KEY = "test-key"
            bridge = LLMBridge()
            with patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": "hello"}}]
                }
                mock_post.return_value = mock_resp
                bridge.query("sys", "user", "MiniMax-M2.7", temperature=0)
                call_args = mock_post.call_args
                payload = call_args[1]["json"]
                self.assertGreater(payload["temperature"], 0.0)
                self.assertLessEqual(payload["temperature"], 1.0)
        finally:
            config.MINIMAX_API_KEY = original_key

    def test_temperature_not_clamped_for_local(self):
        """Local models accept temperature=0; should not be changed."""
        bridge = LLMBridge()
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "hello"}}]
            }
            mock_post.return_value = mock_resp
            bridge.query("sys", "user", "qwen3-vl-8b", temperature=0)
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            self.assertEqual(payload["temperature"], 0)


class TestLLMBridgeGetModels(unittest.TestCase):
    """Unit tests: get_models returns MiniMax models when provider is MiniMax."""

    def test_minimax_provider_returns_minimax_models(self):
        original = config.LLM_PROVIDER
        try:
            config.LLM_PROVIDER = "MiniMax"
            bridge = LLMBridge()
            models = bridge.get_models()
            self.assertIn("MiniMax-M2.7", models)
            self.assertIn("MiniMax-M2.7-highspeed", models)
        finally:
            config.LLM_PROVIDER = original

    def test_lm_studio_provider_queries_local_server(self):
        original = config.LLM_PROVIDER
        try:
            config.LLM_PROVIDER = "LM Studio"
            bridge = LLMBridge()
            with patch("requests.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {
                    "data": [{"id": "local-model-1"}, {"id": "local-model-2"}]
                }
                mock_get.return_value = mock_resp
                models = bridge.get_models()
                self.assertIn("local-model-1", models)
                self.assertNotIn("MiniMax-M2.7", models)
        finally:
            config.LLM_PROVIDER = original

    def test_lm_studio_fallback_when_server_down(self):
        original = config.LLM_PROVIDER
        try:
            config.LLM_PROVIDER = "LM Studio"
            bridge = LLMBridge()
            with patch("requests.get", side_effect=Exception("Connection refused")):
                models = bridge.get_models()
                self.assertTrue(len(models) > 0)
        finally:
            config.LLM_PROVIDER = original


class TestConfigConstants(unittest.TestCase):
    """Unit tests: config constants are correct."""

    def test_minimax_base_url(self):
        self.assertTrue(config.MINIMAX_BASE_URL.startswith("https://api.minimax.io"))

    def test_minimax_models_list(self):
        self.assertIn("MiniMax-M2.7", config.MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.7-highspeed", config.MINIMAX_MODELS)
        self.assertEqual(len(config.MINIMAX_MODELS), 2)

    def test_default_llm_provider(self):
        # Default is LM Studio for backwards compatibility
        self.assertIn(config.LLM_PROVIDER, ["LM Studio", "MiniMax"])


class TestLLMBridgeQueryIntegration(unittest.TestCase):
    """Integration test: actual MiniMax API call (skipped when no API key)."""

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY") or (
            os.path.exists(os.path.expanduser("~/../github_pr/.env.local")) or
            os.path.exists("/home/ximi/github_pr/.env.local")
        ),
        "MINIMAX_API_KEY not available"
    )
    def test_minimax_api_basic_query(self):
        # Try to load key from env file if not in environment
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            env_path = "/home/ximi/github_pr/.env.local"
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("MINIMAX_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
                            break

        if not api_key:
            self.skipTest("No MINIMAX_API_KEY found")

        original_key = config.MINIMAX_API_KEY
        try:
            config.MINIMAX_API_KEY = api_key
            bridge = LLMBridge()
            result = bridge.query(
                "You are a helpful assistant.",
                'Reply with exactly: "MiniMax integration test passed"',
                "MiniMax-M2.7",
                temperature=1.0,
            )
            self.assertIsInstance(result, str)
            self.assertFalse(result.startswith("Error"), f"Query returned error: {result}")
            self.assertTrue(len(result) > 0)
        finally:
            config.MINIMAX_API_KEY = original_key


if __name__ == "__main__":
    unittest.main()
