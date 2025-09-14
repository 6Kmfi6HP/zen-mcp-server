import os
from unittest.mock import patch

from providers.base import ProviderType
from providers.gemini import GeminiModelProvider


class TestGeminiProvider:
    """Test Gemini provider initialization and configuration."""

    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_default(self):
        provider = GeminiModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.GOOGLE
        assert provider.base_url == GeminiModelProvider.DEFAULT_BASE_URL

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_with_custom_url(self):
        provider = GeminiModelProvider("test-key", base_url="https://custom.example.com")
        assert provider.base_url == "https://custom.example.com"

    @patch.dict(os.environ, {"GEMINI_BASE_URL": "https://env.example.com"}, clear=True)
    def test_env_base_url(self):
        provider = GeminiModelProvider("test-key")
        assert provider.base_url == "https://env.example.com"
