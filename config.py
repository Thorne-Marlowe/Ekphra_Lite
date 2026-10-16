from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


def resolve_lora_selection(selection: str | None, directory: str = "models/loras") -> str | None:
    """Resolve a user-selected LoRA filename to a full path safely."""
    if not selection:
        return None
    requested = Path(str(selection)).name.casefold()
    root = Path(directory)
    if not requested or not root.exists():
        return None
    supported = {".safetensors", ".pt", ".ckpt"}
    return next(
        (str(path) for path in root.iterdir()
         if path.is_file() and path.suffix.casefold() in supported
         and path.name.casefold() == requested),
        None,
    )

load_dotenv()

@dataclass
class PathConfig:
    panel_dir: str = "outputs/panels"
    output_page: str = "comic_page.png"

@dataclass
class StoryConfig:
    max_retries: int = 3

@dataclass
class ComfyUIConfig:
    base_url: str = "http://127.0.0.1:8188"
    profiles_file: str = "renderer_profiles.json"
    timeout_seconds: int = 600
    poll_seconds: float = 1.0

@dataclass
class VisionConfig:
    enabled: bool = False

@dataclass
class PanelConfig:
    width: int = 768
    height: int = 1024
    fallback_width: int = 512
    fallback_height: int = 768

@dataclass
class PageConfig:
    width: int = 2480
    height: int = 3508
    margin: int = 60
    gutter: int = 20


@dataclass
class Config:
    paths: PathConfig = None
    story: StoryConfig = None
    page: PageConfig = None
    panel: PanelConfig = None
    comfyui: ComfyUIConfig = None
    vision: VisionConfig = None
    
    def __post_init__(self):
        if self.paths is None:
            self.paths = PathConfig()
        if self.story is None:
            self.story = StoryConfig()
        if self.page is None:
            self.page = PageConfig()
        if self.panel is None:
            self.panel = PanelConfig()
        if self.comfyui is None:
            self.comfyui = ComfyUIConfig()
        if self.vision is None:
            self.vision = VisionConfig()

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

    def from_dict(self, data: dict):
        if 'paths' in data:
            for k, v in data['paths'].items():
                if hasattr(self.paths, k): setattr(self.paths, k, v)
        if 'story' in data:
            for k, v in data['story'].items():
                if hasattr(self.story, k): setattr(self.story, k, v)
        if 'page' in data:
            for k, v in data['page'].items():
                if hasattr(self.page, k): setattr(self.page, k, v)
        if 'panel' in data:
            for k, v in data['panel'].items():
                if hasattr(self.panel, k): setattr(self.panel, k, v)
        if 'comfyui' in data:
            for k, v in data['comfyui'].items():
                if hasattr(self.comfyui, k): setattr(self.comfyui, k, v)
        if 'vision' in data:
            for k, v in data['vision'].items():
                if hasattr(self.vision, k): setattr(self.vision, k, v)

CONFIG = Config()

