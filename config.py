"""
Configuration module for the cognitive architecture ingestion pipeline.

Uses Pydantic BaseSettings to allow configuration via environment variables or a .env file.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration loaded from environment variables or defaults.

    Attributes:
        input_dir: Directory containing input files (book and instructions).
        output_dir: Directory where outputs are written.
        prompts_dir: Directory containing system prompt files for agents.
        model_name: Chat model identifier for the LLM.
        embeddings_model: Name of the sentence-transformers model for embeddings.
    chunk_size: Target chunk size for splitting (tokens when available; falls back to characters).
    chunk_overlap: Overlap between chunks (tokens when available; falls back to characters).
        temperature: LLM temperature.
        max_tokens: Maximum tokens for LLM generation per call (advisory).
        plantuml_cli: Optional path to PlantUML CLI (not required to generate .puml text files).
    """

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    input_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "input")
    output_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "output")
    prompts_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "system_prompts")

    # Models and processing
    model_name: str = Field(default="gpt-4o-mini", description="Chat model name for LLM")
    embeddings_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=150)
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2048)

    # Optional PlantUML CLI path
    plantuml_cli: str | None = None

    # LLM client settings (LM Studio defaults)
    llm_base_url: str = Field(default="http://127.0.0.1:1234/v1")
    llm_api_key: str = Field(default="api_key")
    llm_model: str = Field(default="model")
    llm_temperature: float = Field(default=0.0)
    self_consistency_samples: int = Field(default=3)
    json_repair_retries: int = Field(default=2)

    # Text normalization controls
    normalize_whitespace: bool = Field(default=True)
    transliterate_text: bool = Field(default=True)

    # Ingestion controls
    full_ingestion: bool = Field(default=False)
    full_ingestion_batch_size: int = Field(default=5)
    reuse_full_ingestion_cache: bool = Field(default=True)
    # Resume controls
    resume_from_cache: bool = Field(default=False)
    resume_allow_hash_mismatch: bool = Field(default=False)
    resume_require_full_ingestion: bool = Field(default=True)
    resume_skip_pipeline_if_complete: bool = Field(default=False)

    # Vector index persistence
    persist_vector_index: bool = Field(default=True)
    reuse_vector_index: bool = Field(default=True)

    # Diagram generation
    diagram_format: str = Field(default="plantuml", description="plantuml or dot")
    graphviz_render: bool = Field(default=False, description="Render DOT to image if Graphviz available")
    graphviz_engine: str = Field(default="dot", description="Graphviz layout engine (dot, neato, fdp, etc.)")

    # Graph building (IO linking)
    io_link_similarity_threshold: float = Field(default=0.62)
    max_candidate_edges: int = Field(default=200)

    # Deduper controls
    name_similarity_threshold: float = Field(default=0.9)
    use_llm_deduper: bool = Field(default=False)
    llm_similarity_band: float = Field(default=0.1, description="Jaccard band below threshold where LLM is consulted")
    llm_deduper_max_checks: int = Field(default=25, description="Cap LLM comparisons to limit calls")

    # PlantUML rendering via server (optional)
    plantuml_server_url: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("input_dir", "output_dir", "prompts_dir", pre=True)
    def _ensure_path(cls, v):  # type: ignore[override]
        return Path(v) if not isinstance(v, Path) else v


settings = Config()
