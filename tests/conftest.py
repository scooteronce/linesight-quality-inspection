import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "web"))

# main.py imports the Anthropic SDK and AgentCore at module level; neither is
# installed locally (they are packaged into the runtime bundle), so stub them
# before any test module imports main.
anthropic_stub = types.ModuleType("anthropic")
anthropic_stub.AnthropicBedrock = MagicMock()
anthropic_stub.AnthropicBedrockMantle = MagicMock()
sys.modules.setdefault("anthropic", anthropic_stub)


class _FakeAgentCoreApp:
    def entrypoint(self, func):
        return func

    def run(self):
        pass


bedrock_agentcore_stub = types.ModuleType("bedrock_agentcore")
runtime_stub = types.ModuleType("bedrock_agentcore.runtime")
runtime_stub.BedrockAgentCoreApp = _FakeAgentCoreApp
sys.modules.setdefault("bedrock_agentcore", bedrock_agentcore_stub)
sys.modules.setdefault("bedrock_agentcore.runtime", runtime_stub)
