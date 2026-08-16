import unittest
from unittest.mock import Mock, patch

from models import LLMBridge


class LLMBridgeTests(unittest.TestCase):
    def test_normalises_root_and_adds_bearer_header(self):
        bridge = LLMBridge("http://192.168.2.192:1234/v1", "secret-token")
        self.assertEqual(bridge.root_url, "http://192.168.2.192:1234")
        self.assertEqual(bridge.base_url, "http://192.168.2.192:1234/v1")
        self.assertEqual(bridge._headers()["Authorization"], "Bearer secret-token")

    @patch("models.requests.get")
    def test_lists_only_vision_capable_llms(self, mock_get):
        response = Mock(status_code=200)
        response.json.return_value = {
            "models": [
                {"type": "llm", "key": "vision/model", "capabilities": {"vision": True}},
                {"type": "llm", "key": "text/model", "capabilities": {"vision": False}},
                {"type": "embedding", "key": "embed/model"},
            ]
        }
        mock_get.return_value = response
        bridge = LLMBridge("http://localhost:1234", "token")
        self.assertEqual(bridge.get_models(), ["vision/model"])
        mock_get.assert_called_once_with(
            "http://localhost:1234/api/v1/models",
            headers={"Content-Type": "application/json", "Authorization": "Bearer token"},
            timeout=10,
        )

    @patch("models.requests.post")
    def test_query_uses_openai_compatible_path_and_auth(self, mock_post):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "rewritten"}}]}
        mock_post.return_value = response
        bridge = LLMBridge("http://localhost:1234", "token")
        self.assertEqual(bridge.query("system", "user", "vision/model"), "rewritten")
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:1234/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer token")


if __name__ == "__main__":
    unittest.main()
