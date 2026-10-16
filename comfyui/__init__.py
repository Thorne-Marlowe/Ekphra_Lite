"""ComfyUI integration boundary for the comic renderer."""

from .client import ComfyUIClient, ComfyUIError
from .profiles import RendererProfile, RendererProfileStore

__all__ = ["ComfyUIClient", "ComfyUIError", "RendererProfile", "RendererProfileStore"]
