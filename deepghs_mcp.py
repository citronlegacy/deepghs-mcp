"""
DeepGHS MCP Server
Provides tools for discovering and working with DeepGHS datasets, models, and spaces
on HuggingFace, plus code generation for anime AI training pipelines.

DeepGHS (Deep Generative anime Hobbyist Syndicate) is a non-profit open-source
community building anime/2D-focused AI infrastructure. Their HuggingFace org
contains Danbooru/Sankaku/Gelbooru/Zerochan full datasets, character image
collections, detection/classification models, and the waifuc/cheesechaser tooling.
"""

import json
import os
import textwrap
from enum import Enum
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

HF_API_BASE = "https://huggingface.co/api"
DEEPGHS_ORG = "deepghs"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

DEFAULT_LIMIT = 20
MAX_LIMIT = 100

# Known DeepGHS dataset categories for discoverability
DATASET_CATEGORIES = {
    "booru": ["danbooru2024", "sankaku", "gelbooru", "yande", "rule34", "konachan", "anime-pictures", "zerochan"],
    "character": ["bangumibase", "character", "similarity"],
    "detection": ["face-detection", "head-detection", "anime-face"],
    "tags": ["site_tags", "tags"],
    "functional": ["ccip", "aesthetic", "classifier"],
}

# ─────────────────────────────────────────────────────────────
# Server Init
# ─────────────────────────────────────────────────────────────

mcp = FastMCP("deepghs_mcp")

# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"

class SortBy(str, Enum):
    DOWNLOADS = "downloads"
    LIKES = "likes"
    CREATED = "createdAt"
    MODIFIED = "lastModified"

class ModelFormat(str, Enum):
    SD15 = "sd1.5"
    SDXL = "sdxl"
    FLUX = "flux"

class ImageSource(str, Enum):
    DANBOORU = "danbooru"
    PIXIV = "pixiv"
    GELBOORU = "gelbooru"
    ZEROCHAN = "zerochan"
    SANKAKU = "sankaku"
    AUTO = "auto"

class ContentRating(str, Enum):
    SAFE = "safe"
    SAFE_R15 = "safe_r15"
    ALL = "all"

# ─────────────────────────────────────────────────────────────
# HTTP Client Helper
# ─────────────────────────────────────────────────────────────

def _auth_headers() -> dict:
    """Build authorization headers if HF_TOKEN is set."""
    if HF_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}


async def hf_get(path: str, params: Optional[dict] = None) -> dict | list:
    """
    Perform a GET request to the HuggingFace Hub API.

    Args:
        path: API path, e.g. '/datasets' or '/datasets/deepghs/danbooru2024'
        params: Optional query string parameters

    Returns:
        Parsed JSON (dict or list)

    Raises:
        httpx.HTTPStatusError: On non-2xx responses
        httpx.TimeoutException: On timeout
    """
    url = f"{HF_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, params=params or {}, headers=_auth_headers())
        response.raise_for_status()
        return response.json()


def handle_error(e: Exception) -> str:
    """Return a consistent, actionable error message."""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 401:
            return "Error 401: Unauthorized. Set HF_TOKEN in your MCP env config for private/gated repos."
        if code == 403:
            return "Error 403: Forbidden. This resource may be gated — request access at huggingface.co first."
        if code == 404:
            return "Error 404: Not found. Check the repo_id is correct (e.g. 'deepghs/danbooru2024')."
        if code == 429:
            return "Error 429: Rate limited. Set HF_TOKEN in your MCP config to get higher rate limits."
        return f"Error {code}: HuggingFace API error — {e.response.text[:300]}"
    if isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. HuggingFace may be slow. Try again shortly."
    return f"Error: {type(e).__name__}: {str(e)}"

# ─────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────

def _fmt_size(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def _fmt_dataset_card(d: dict) -> str:
    repo_id = d.get("id", "N/A")
    downloads = _fmt_size(d.get("downloads"))
    likes = d.get("likes", 0)
    last_mod = d.get("lastModified", "N/A")[:10]
    tags = d.get("tags", [])
    sha = d.get("sha", "")[:8]
    url = f"https://huggingface.co/datasets/{repo_id}"
    tag_str = ", ".join(tags[:8]) + (f" (+{len(tags)-8} more)" if len(tags) > 8 else "")
    return (
        f"**[{repo_id}]({url})**\n"
        f"  Downloads: {downloads} | Likes: {likes} | Updated: {last_mod}\n"
        f"  Tags: {tag_str or 'none'}\n"
        f"  SHA: `{sha}`"
    )


def _fmt_model_card(m: dict) -> str:
    repo_id = m.get("id", "N/A")
    downloads = _fmt_size(m.get("downloads"))
    likes = m.get("likes", 0)
    pipeline = m.get("pipeline_tag", "N/A")
    last_mod = m.get("lastModified", "N/A")[:10]
    url = f"https://huggingface.co/models/{repo_id}"
    return (
        f"**[{repo_id}]({url})**\n"
        f"  Task: {pipeline} | Downloads: {downloads} | Likes: {likes} | Updated: {last_mod}"
    )


def _fmt_space_card(s: dict) -> str:
    repo_id = s.get("id", "N/A")
    likes = s.get("likes", 0)
    sdk = s.get("sdk", "N/A")
    last_mod = s.get("lastModified", "N/A")[:10]
    url = f"https://huggingface.co/spaces/{repo_id}"
    return (
        f"**[{repo_id}]({url})**\n"
        f"  SDK: {sdk} | Likes: {likes} | Updated: {last_mod}"
    )


def _fmt_file_tree(siblings: list) -> str:
    if not siblings:
        return "No files found."
    lines = []
    total_size = 0
    for s in siblings:
        fname = s.get("rfilename", "")
        size = s.get("size")
        blob_id = s.get("blobId", "")[:8]
        size_str = _fmt_size(size) + "B" if size else "?"
        if size:
            total_size += size
        lines.append(f"  `{fname}` — {size_str} ({blob_id})")
    lines.append(f"\n  **Total: {_fmt_size(total_size)}B across {len(siblings)} files**")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Input Models
# ─────────────────────────────────────────────────────────────

class ListDatasetsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    search: Optional[str] = Field(
        default=None,
        description="Keyword to filter datasets by name (e.g. 'danbooru', 'character', 'face', 'tags')",
        max_length=100
    )
    sort: SortBy = Field(
        default=SortBy.DOWNLOADS,
        description="Sort order: 'downloads', 'likes', 'createdAt', or 'lastModified'"
    )
    limit: int = Field(default=DEFAULT_LIMIT, description="Max results to return (1–100)", ge=1, le=MAX_LIMIT)
    offset: int = Field(default=0, description="Offset for pagination", ge=0)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


class ListModelsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    search: Optional[str] = Field(
        default=None,
        description="Keyword to filter models (e.g. 'ccip', 'aesthetic', 'tagger', 'face', 'classifier')",
        max_length=100
    )
    sort: SortBy = Field(default=SortBy.DOWNLOADS, description="Sort order: 'downloads', 'likes', 'createdAt', 'lastModified'")
    limit: int = Field(default=DEFAULT_LIMIT, description="Max results to return (1–100)", ge=1, le=MAX_LIMIT)
    offset: int = Field(default=0, description="Offset for pagination", ge=0)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


class ListSpacesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    search: Optional[str] = Field(
        default=None,
        description="Keyword to filter spaces (e.g. 'detection', 'tagger', 'search', 'demo')",
        max_length=100
    )
    limit: int = Field(default=DEFAULT_LIMIT, description="Max results to return (1–100)", ge=1, le=MAX_LIMIT)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


class GetRepoInfoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    repo_id: str = Field(
        ...,
        description="Full HuggingFace repo ID (e.g. 'deepghs/danbooru2024', 'deepghs/site_tags')",
        min_length=3, max_length=200
    )
    repo_type: str = Field(
        default="dataset",
        description="Type of repo: 'dataset', 'model', or 'space'",
        pattern="^(dataset|model|space)$"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


class SearchTagsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    tag: str = Field(
        ...,
        description="Tag to look up across booru platforms (e.g. 'hatsune_miku', 'Hatsune Miku', '初音ミク'). Any format/language accepted.",
        min_length=1, max_length=200
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


class FindCharacterDatasetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    character_name: str = Field(
        ...,
        description="Character name to search for (e.g. 'Rem', 'Hatsune Miku', 'surtr arknights'). Will search both deepghs and CyberHarem namespaces.",
        min_length=1, max_length=200
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


class GenerateWaifucScriptInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    character_name: str = Field(
        ...,
        description="Character name for the dataset (e.g. 'Rem', 'surtr arknights', 'hatsune miku')",
        min_length=1, max_length=200
    )
    danbooru_tag: Optional[str] = Field(
        default=None,
        description="Danbooru tag for the character (snake_case, e.g. 'rem_(re:zero)', 'surtr_(arknights)'). If omitted, a best-guess is used.",
        max_length=200
    )
    pixiv_query: Optional[str] = Field(
        default=None,
        description="Pixiv search query for the character, in Japanese or English (e.g. 'レム', 'アークナイツ スルト'). Required if pixiv is in sources.",
        max_length=200
    )
    sources: list[ImageSource] = Field(
        default=[ImageSource.DANBOORU],
        description="Image sources to crawl: 'danbooru', 'pixiv', 'gelbooru', 'zerochan', 'sankaku', or 'auto' (uses GcharAutoSource for game characters)"
    )
    model_format: ModelFormat = Field(
        default=ModelFormat.SD15,
        description="Target model format: 'sd1.5', 'sdxl', or 'flux'. Controls crop size and exporter settings."
    )
    content_rating: ContentRating = Field(
        default=ContentRating.SAFE,
        description="Content rating filter: 'safe', 'safe_r15' (includes mild ecchi), or 'all' (no filter)"
    )
    output_dir: str = Field(
        default="./dataset_output",
        description="Output directory for the crawled dataset",
        max_length=300
    )
    max_images: Optional[int] = Field(
        default=None,
        description="Maximum number of images to collect (e.g. 500). If None, collects all available.",
        ge=1, le=50000
    )
    pixiv_token: Optional[str] = Field(
        default=None,
        description="Pixiv refresh token (required if pixiv is in sources). Get yours from pixivpy documentation.",
        max_length=200
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list) -> list:
        if not v:
            raise ValueError("At least one source is required.")
        return list(dict.fromkeys(v))  # deduplicate


class GenerateCheesechaserScriptInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    repo_id: str = Field(
        ...,
        description="HuggingFace dataset repo ID to download from (e.g. 'deepghs/danbooru2024')",
        min_length=3, max_length=200
    )
    output_dir: str = Field(
        default="./downloads",
        description="Local directory to save downloaded files",
        max_length=300
    )
    post_ids: Optional[list[int]] = Field(
        default=None,
        description="Specific post/image IDs to download selectively. If None, downloads all available (use with caution on large datasets).",
    )
    max_workers: int = Field(
        default=4,
        description="Number of parallel download workers (1–16)",
        ge=1, le=16
    )


# ─────────────────────────────────────────────────────────────
# Tools — Discovery
# ─────────────────────────────────────────────────────────────

@mcp.tool(
    name="deepghs_list_datasets",
    annotations={
        "title": "List DeepGHS Datasets on HuggingFace",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def deepghs_list_datasets(params: ListDatasetsInput) -> str:
    """List all public datasets from the DeepGHS organization on HuggingFace.

    DeepGHS publishes datasets including Danbooru2024 (8M+ images), Sankaku,
    Gelbooru, Zerochan, BangumiBase (character frames), site_tags (cross-platform
    tag database), face/head detection datasets, and more.

    Args:
        params (ListDatasetsInput):
            - search (Optional[str]): Keyword filter (e.g. 'danbooru', 'character', 'face')
            - sort (SortBy): Sort by 'downloads', 'likes', 'createdAt', 'lastModified'
            - limit (int): Results per page, 1–100 (default: 20)
            - offset (int): Pagination offset (default: 0)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Paginated list of datasets with download counts, likes, update dates,
             tags, and direct HuggingFace links.
    """
    query: dict = {
        "author": DEEPGHS_ORG,
        "sort": params.sort.value,
        "limit": params.limit,
        "full": "false",
    }
    if params.search:
        query["search"] = params.search

    try:
        data = await hf_get("/datasets", query)
    except Exception as e:
        return handle_error(e)

    if not isinstance(data, list):
        data = data.get("datasets", []) if isinstance(data, dict) else []

    # Apply offset manually (HF API doesn't always support it for org queries)
    paged = data[params.offset: params.offset + params.limit]

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "total": len(data),
            "count": len(paged),
            "offset": params.offset,
            "has_more": params.offset + params.limit < len(data),
            "next_offset": params.offset + params.limit if params.offset + params.limit < len(data) else None,
            "datasets": paged,
        }, indent=2, ensure_ascii=False)

    if not paged:
        return f"No datasets found for query: `{params.search or 'all'}`"

    lines = [
        f"## DeepGHS Datasets — `{params.search or 'all'}` ({len(paged)} of {len(data)} results)\n",
        f"Sorted by: **{params.sort.value}** | Page offset: {params.offset}\n",
    ]
    for d in paged:
        lines.append(_fmt_dataset_card(d))
        lines.append("")

    if params.offset + params.limit < len(data):
        lines.append(f"*Use `offset={params.offset + params.limit}` to see the next page.*")

    return "\n".join(lines)


@mcp.tool(
    name="deepghs_list_models",
    annotations={
        "title": "List DeepGHS Models on HuggingFace",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def deepghs_list_models(params: ListModelsInput) -> str:
    """List all public models from the DeepGHS organization on HuggingFace.

    DeepGHS models include: CCIP (character similarity encoder), WD Tagger Enhanced
    (anime image tagger with embeddings), aesthetic scorer, anime/real classifier,
    image type classifier, furry detector, face/head/person detection models,
    NSFW censor, and style era classifier.

    Args:
        params (ListModelsInput):
            - search (Optional[str]): Keyword filter (e.g. 'ccip', 'tagger', 'aesthetic', 'face')
            - sort (SortBy): Sort by 'downloads', 'likes', 'createdAt', 'lastModified'
            - limit (int): Results per page, 1–100 (default: 20)
            - offset (int): Pagination offset (default: 0)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Paginated list of models with task type, download counts, likes, and links.
    """
    query: dict = {
        "author": DEEPGHS_ORG,
        "sort": params.sort.value,
        "limit": params.limit,
    }
    if params.search:
        query["search"] = params.search

    try:
        data = await hf_get("/models", query)
    except Exception as e:
        return handle_error(e)

    if not isinstance(data, list):
        data = []

    paged = data[params.offset: params.offset + params.limit]

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "total": len(data),
            "count": len(paged),
            "offset": params.offset,
            "has_more": params.offset + params.limit < len(data),
            "models": paged,
        }, indent=2, ensure_ascii=False)

    if not paged:
        return f"No models found for query: `{params.search or 'all'}`"

    lines = [f"## DeepGHS Models — `{params.search or 'all'}` ({len(paged)} of {len(data)} results)\n"]
    for m in paged:
        lines.append(_fmt_model_card(m))
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="deepghs_list_spaces",
    annotations={
        "title": "List DeepGHS Spaces (Live Demos) on HuggingFace",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def deepghs_list_spaces(params: ListSpacesInput) -> str:
    """List all public Spaces (live demo apps) from the DeepGHS organization on HuggingFace.

    DeepGHS spaces include: reverse image search, Danbooru character lookup,
    anime face/head/person detection demos, CCIP character similarity demo,
    WD tagger demo, aesthetic scorer demo, and more.

    Args:
        params (ListSpacesInput):
            - search (Optional[str]): Keyword filter (e.g. 'detection', 'tagger', 'search')
            - limit (int): Results per page, 1–100 (default: 20)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: List of Spaces with SDK type, likes, update dates, and direct links.
    """
    query: dict = {
        "author": DEEPGHS_ORG,
        "limit": params.limit,
    }
    if params.search:
        query["search"] = params.search

    try:
        data = await hf_get("/spaces", query)
    except Exception as e:
        return handle_error(e)

    if not isinstance(data, list):
        data = []

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"count": len(data), "spaces": data}, indent=2, ensure_ascii=False)

    if not data:
        return f"No spaces found for query: `{params.search or 'all'}`"

    lines = [f"## DeepGHS Spaces — `{params.search or 'all'}` ({len(data)} results)\n"]
    for s in data:
        lines.append(_fmt_space_card(s))
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="deepghs_get_repo_info",
    annotations={
        "title": "Get Full Info for a DeepGHS Dataset, Model, or Space",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def deepghs_get_repo_info(params: GetRepoInfoInput) -> str:
    """Get detailed metadata for a specific DeepGHS dataset, model, or space by repo ID.

    Returns the full file tree with sizes, all tags, README card metadata,
    download counts, creation/modification dates, and gating status.
    Use this before deciding to download — the file tree shows you exactly
    what tar/parquet files are inside and how large they are.

    Args:
        params (GetRepoInfoInput):
            - repo_id (str): Full HF repo ID (e.g. 'deepghs/danbooru2024')
            - repo_type (str): 'dataset', 'model', or 'space' (default: 'dataset')
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Full repo metadata including file tree with sizes, tags, card data,
             and a generated cheesechaser download command if applicable.

    Schema (JSON mode):
        {
            "id": str,
            "sha": str,
            "lastModified": str,
            "tags": list[str],
            "downloads": int,
            "likes": int,
            "cardData": dict,         # README metadata
            "siblings": [             # File tree
                {"rfilename": str, "size": int, "blobId": str}
            ],
            "gated": bool | str
        }
    """
    type_path = "datasets" if params.repo_type == "dataset" else (
        "models" if params.repo_type == "model" else "spaces"
    )
    try:
        data = await hf_get(f"/{type_path}/{params.repo_id}", {"full": "true"})
    except Exception as e:
        return handle_error(e)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, indent=2, ensure_ascii=False)

    repo_id = data.get("id", params.repo_id)
    sha = data.get("sha", "")[:8]
    last_mod = data.get("lastModified", "N/A")[:10]
    created = data.get("createdAt", "N/A")[:10]
    downloads = _fmt_size(data.get("downloads"))
    likes = data.get("likes", 0)
    tags = data.get("tags", [])
    gated = data.get("gated", False)
    siblings = data.get("siblings", [])
    card = data.get("cardData", {}) or {}

    url = f"https://huggingface.co/{type_path}/{repo_id}"

    lines = [
        f"## [{repo_id}]({url})",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Type | {params.repo_type} |",
        f"| Created | {created} |",
        f"| Updated | {last_mod} |",
        f"| SHA | `{sha}` |",
        f"| Downloads | {downloads} |",
        f"| Likes | {likes} |",
        f"| Gated | {'⚠️ Yes — request access first' if gated else '✅ No'} |",
        f"| Files | {len(siblings)} |",
        f"",
        f"**Tags:** {', '.join(tags[:15]) or 'none'}",
        f"",
    ]

    if card:
        license_ = card.get("license", "N/A")
        task = card.get("task_categories", [])
        lines += [
            f"**License:** {license_}",
            f"**Tasks:** {', '.join(task) if task else 'N/A'}",
            f"",
        ]

    if siblings:
        lines.append(f"### File Tree ({len(siblings)} files)")
        lines.append(_fmt_file_tree(siblings))
        lines.append("")

    # Suggest cheesechaser command for indexed datasets
    if params.repo_type == "dataset":
        has_tar = any(s.get("rfilename", "").endswith(".tar") for s in siblings)
        if has_tar:
            lines += [
                f"### 💡 Download with cheesechaser",
                f"```python",
                f"from cheesechaser.datapool import SimpleDataPool",
                f"pool = SimpleDataPool('{repo_id}')",
                f"pool.batch_download_to_directory(",
                f"    resource_ids=[...],  # list of post IDs, or omit for all",
                f"    local_directory='./downloads',",
                f"    max_workers=4,",
                f")",
                f"```",
            ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Tools — Tag Intelligence
# ─────────────────────────────────────────────────────────────

@mcp.tool(
    name="deepghs_search_tags",
    annotations={
        "title": "Search Tag Cross-Platform Reference (site_tags)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def deepghs_search_tags(params: SearchTagsInput) -> str:
    """Search the DeepGHS site_tags dataset — the definitive cross-platform anime tag reference.

    The deepghs/site_tags dataset covers 2.5M+ unique tags across 18 platforms:
    Danbooru, Gelbooru, Pixiv, Sankaku, Wallhaven, Yande.re, Konachan, Zerochan,
    Rule34, and more. Each tag has category, post count, and aliases per platform.

    This is the key tool for the MultiBoru tag normalization problem:
    - Danbooru uses snake_case:   hatsune_miku
    - Zerochan uses Title Case:   Hatsune Miku
    - Pixiv uses Japanese:        初音ミク
    This tool maps them all together.

    Args:
        params (SearchTagsInput):
            - tag (str): Tag in any format/language/platform
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Cross-platform tag information including canonical names per platform,
             post counts, tag category (character/copyright/artist/general),
             and known aliases. Includes direct dataset link for full data access.

    Note:
        This tool returns the HuggingFace dataset info and provides query guidance
        for the site_tags dataset. For programmatic tag lookup at scale, use the
        dataset's Parquet or SQLite files directly via the dataset viewer API.
    """
    # Return structured guidance + dataset API link for the tag
    tag_encoded = params.tag.replace(" ", "%20")
    dataset_viewer_url = (
        f"https://datasets-server.huggingface.co/search"
        f"?dataset=deepghs%2Fsite_tags&config=default&split=train"
        f"&query={tag_encoded}&offset=0&length=10"
    )
    dataset_url = "https://huggingface.co/datasets/deepghs/site_tags"

    # Platform-specific format guidance
    platforms = {
        "Danbooru":    "snake_case — e.g. `hatsune_miku`, `rem_(re:zero)`",
        "Gelbooru":    "snake_case — same as Danbooru in most cases",
        "Sankaku":     "snake_case with category prefix — e.g. `character:hatsune_miku`",
        "Zerochan":    "Title Case — e.g. `Hatsune Miku`, `Rem`",
        "Pixiv":       "Japanese preferred — e.g. `初音ミク`, `レム`",
        "Yande.re":    "snake_case — similar to Danbooru",
        "Konachan":    "snake_case — similar to Danbooru",
        "Rule34":      "snake_case — similar to Danbooru",
        "Wallhaven":   "natural language — e.g. `hatsune miku`",
    }

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "query": params.tag,
            "dataset": "deepghs/site_tags",
            "dataset_url": dataset_url,
            "viewer_search_url": dataset_viewer_url,
            "platform_formats": platforms,
            "note": "Use the viewer_search_url to find exact tag matches across all platforms in the site_tags dataset.",
        }, indent=2)

    lines = [
        f"## Tag Lookup: `{params.tag}`",
        f"",
        f"**Dataset:** [deepghs/site_tags]({dataset_url})",
        f"2.5M+ tags unified across 18 platforms — Danbooru, Gelbooru, Pixiv, Sankaku, Zerochan, etc.",
        f"",
        f"### 🔍 Search this tag in site_tags",
        f"[Click to search `{params.tag}` in dataset viewer]({dataset_viewer_url})",
        f"",
        f"### Platform Tag Format Reference",
        f"Different platforms use different tag formats for the same character:",
        f"",
    ]
    for platform, fmt in platforms.items():
        lines.append(f"- **{platform}**: {fmt}")

    lines += [
        f"",
        f"### Using site_tags Programmatically",
        f"The dataset ships as **Parquet, CSV, JSON, and SQLite**. For bulk tag normalization:",
        f"```python",
        f"import pandas as pd",
        f"# Load the Parquet file (fastest for large queries)",
        f"df = pd.read_parquet('hf://datasets/deepghs/site_tags/data/train-*.parquet')",
        f"# Find your tag",
        f"matches = df[df['name'].str.contains('{params.tag}', case=False)]",
        f"print(matches[['name', 'site', 'category', 'post_count']].head(20))",
        f"```",
        f"",
        f"### Key Tag Categories",
        f"- `character` — specific anime characters",
        f"- `copyright` — series/franchise names",
        f"- `artist` — artist names",
        f"- `general` — descriptive tags (hair color, pose, etc.)",
        f"- `meta` — metadata tags (resolution, file type)",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Tools — Character Dataset Finder
# ─────────────────────────────────────────────────────────────

@mcp.tool(
    name="deepghs_find_character_dataset",
    annotations={
        "title": "Find Pre-Built Character Dataset for LoRA Training",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def deepghs_find_character_dataset(params: FindCharacterDatasetInput) -> str:
    """Search for pre-built character image datasets for LoRA training on HuggingFace.

    Searches both deepghs (BangumiBase) and CyberHarem namespaces for datasets
    built around a specific character. These are pre-crawled, pre-cleaned, and
    ready to use — saving you from having to run waifuc yourself.

    CyberHarem datasets are built using the full DeepGHS automated pipeline:
    crawl → face filter → CCIP identity filter → WD tag → upload to HF.

    Args:
        params (FindCharacterDatasetInput):
            - character_name (str): Character name to search for (e.g. 'Rem', 'Hatsune Miku')
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: List of matching character datasets with image counts, sources,
             download commands, and links. Also suggests waifuc script generation
             if no pre-built dataset is found.
    """
    search_term = params.character_name.lower().replace(" ", "_")
    search_term_space = params.character_name.lower()

    results = []
    errors = []

    # Search deepghs namespace
    for query_str in [search_term, search_term_space]:
        try:
            data = await hf_get("/datasets", {
                "author": DEEPGHS_ORG,
                "search": query_str,
                "limit": 10,
            })
            if isinstance(data, list):
                results.extend(data)
        except Exception as e:
            errors.append(f"deepghs search error: {str(e)[:100]}")
        break

    # Search CyberHarem namespace
    try:
        cyberharem_data = await hf_get("/datasets", {
            "author": "CyberHarem",
            "search": search_term,
            "limit": 15,
        })
        if isinstance(cyberharem_data, list):
            for item in cyberharem_data:
                item["_source_org"] = "CyberHarem"
            results.extend(cyberharem_data)
    except Exception as e:
        errors.append(f"CyberHarem search error: {str(e)[:100]}")

    # Deduplicate by id
    seen = set()
    unique_results = []
    for r in results:
        rid = r.get("id", "")
        if rid not in seen:
            seen.add(rid)
            unique_results.append(r)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "query": params.character_name,
            "count": len(unique_results),
            "results": unique_results,
            "errors": errors,
        }, indent=2, ensure_ascii=False)

    lines = [
        f"## Character Dataset Search: `{params.character_name}`",
        f"",
    ]

    if not unique_results:
        lines += [
            f"❌ No pre-built datasets found for `{params.character_name}`.",
            f"",
            f"### Next Steps",
            f"1. Use **`deepghs_generate_waifuc_script`** to generate a crawling script",
            f"2. Run it to build your own dataset from Danbooru/Pixiv/etc.",
            f"3. Or check manually at: https://huggingface.co/CyberHarem",
        ]
        if errors:
            lines.append(f"\n*Search errors: {'; '.join(errors)}*")
        return "\n".join(lines)

    lines.append(f"Found **{len(unique_results)}** dataset(s):\n")

    for d in unique_results:
        repo_id = d.get("id", "N/A")
        org = d.get("_source_org", "deepghs")
        downloads = _fmt_size(d.get("downloads"))
        likes = d.get("likes", 0)
        last_mod = d.get("lastModified", "N/A")[:10]
        url = f"https://huggingface.co/datasets/{repo_id}"

        lines += [
            f"### [{repo_id}]({url})",
            f"Org: `{org}` | Downloads: {downloads} | Likes: {likes} | Updated: {last_mod}",
            f"",
            f"**Download with cheesechaser:**",
            f"```python",
            f"from cheesechaser.datapool import SimpleDataPool",
            f"pool = SimpleDataPool('{repo_id}')",
            f"pool.batch_download_to_directory('./dataset', max_workers=4)",
            f"```",
            f"",
        ]

    lines += [
        f"---",
        f"💡 Use **`deepghs_get_repo_info`** with any of these repo IDs to see the full file tree and sizes.",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Tools — Training Pipeline Code Generation
# ─────────────────────────────────────────────────────────────

@mcp.tool(
    name="deepghs_generate_waifuc_script",
    annotations={
        "title": "Generate waifuc Data Collection Script for LoRA Training",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def deepghs_generate_waifuc_script(params: GenerateWaifucScriptInput) -> str:
    """Generate a ready-to-run waifuc Python script to crawl and clean anime character images for LoRA training.

    waifuc is DeepGHS's data pipeline framework. This tool generates a complete,
    properly-configured script that:
      1. Crawls images from the specified sources (Danbooru, Pixiv, Gelbooru, etc.)
      2. Converts to RGB and standardizes backgrounds
      3. Filters monochrome/sketch/3D images (NoMonochromeAction, ClassFilterAction)
      4. Filters duplicate/similar images (FilterSimilarAction)
      5. Detects and splits to single-person crops (FaceCountAction, PersonSplitAction)
      6. Filters out wrong characters using CCIP AI identity matching (CCIPAction)
      7. Tags all images with WD14 tagger (TaggingAction)
      8. Crops to target resolution for the specified model format (SD1.5/SDXL/Flux)
      9. Exports in the correct format for the target trainer

    Crop sizes by model format:
      - SD1.5: 512×512 base, bucket range 256–768
      - SDXL:  1024×1024 base, bucket range 512–2048
      - Flux:  1024×1024 base, bucket range 512–2048

    Args:
        params (GenerateWaifucScriptInput):
            - character_name (str): Character display name (used in comments/output path)
            - danbooru_tag (Optional[str]): Danbooru tag e.g. 'rem_(re:zero)'
            - pixiv_query (Optional[str]): Pixiv search string e.g. 'レム リゼロ'
            - sources (list[ImageSource]): ['danbooru', 'pixiv', 'gelbooru', 'zerochan', 'sankaku', 'auto']
            - model_format (ModelFormat): 'sd1.5', 'sdxl', or 'flux'
            - content_rating (ContentRating): 'safe', 'safe_r15', or 'all'
            - output_dir (str): Output directory path
            - max_images (Optional[int]): Max images to collect
            - pixiv_token (Optional[str]): Pixiv refresh token (required for Pixiv source)

    Returns:
        str: Complete, ready-to-run Python script with inline comments explaining
             each pipeline action and its purpose for LoRA training quality.
    """
    char = params.character_name
    char_safe = char.lower().replace(" ", "_").replace("(", "").replace(")", "")
    danbooru_tag = params.danbooru_tag or f"{char_safe}"
    output_dir = params.output_dir.rstrip("/")

    # Determine crop size based on model format
    if params.model_format == ModelFormat.SD15:
        min_size = 512
        pad_size = 512
        crop_note = "512×512 (SD 1.5 standard)"
    elif params.model_format in (ModelFormat.SDXL, ModelFormat.FLUX):
        min_size = 1024
        pad_size = 1024
        crop_note = "1024×1024 (SDXL/Flux standard)"

    # Content rating filter
    if params.content_rating == ContentRating.SAFE:
        rating_filter_code = "    RatingFilterAction(['safe']),  # drop anything above safe rating"
        rating_import = "RatingFilterAction, "
    elif params.content_rating == ContentRating.SAFE_R15:
        rating_filter_code = "    RatingFilterAction(['safe', 'r15']),  # allow safe + mild ecchi"
        rating_import = "RatingFilterAction, "
    else:
        rating_filter_code = "    # RatingFilterAction disabled — all ratings included"
        rating_import = ""

    # Build source blocks
    source_imports = []
    source_blocks = []

    for src in params.sources:
        if src == ImageSource.DANBOORU:
            source_imports.append("DanbooruSource")
            source_blocks.append(textwrap.dedent(f"""\
                # ── Danbooru Source ──────────────────────────────────────────────────
                # Tip: Adding 'solo' tag reduces multi-character images upstream,
                # but PersonSplitAction below handles them anyway.
                danbooru_source = DanbooruSource(['{danbooru_tag}'])
            """))

        elif src == ImageSource.PIXIV:
            pixiv_q = params.pixiv_query or char
            token_val = params.pixiv_token or "YOUR_PIXIV_REFRESH_TOKEN"
            source_imports.append("PixivSearchSource")
            source_blocks.append(textwrap.dedent(f"""\
                # ── Pixiv Source ─────────────────────────────────────────────────────
                # Get your refresh token: https://github.com/upbit/pixivpy
                # CCIP is especially important for Pixiv — lots of irrelevant characters
                pixiv_source = PixivSearchSource(
                    '{pixiv_q}',
                    refresh_token='{token_val}',
                )
            """))

        elif src == ImageSource.GELBOORU:
            source_imports.append("GelbooruSource")
            source_blocks.append(textwrap.dedent(f"""\
                # ── Gelbooru Source ──────────────────────────────────────────────────
                # Gelbooru supports many tags simultaneously (unlike Danbooru's 2-tag limit)
                gelbooru_source = GelbooruSource(['{danbooru_tag}'])
            """))

        elif src == ImageSource.ZEROCHAN:
            zc_tag = char  # Zerochan uses Title Case
            source_imports.append("ZerochanSource")
            source_blocks.append(textwrap.dedent(f"""\
                # ── Zerochan Source ──────────────────────────────────────────────────
                # Zerochan uses Title Case tags — typically higher quality art
                zerochan_source = ZerochanSource('{zc_tag}', strict=True)
            """))

        elif src == ImageSource.SANKAKU:
            source_imports.append("SankakuSource")
            source_blocks.append(textwrap.dedent(f"""\
                # ── Sankaku Source ───────────────────────────────────────────────────
                sankaku_source = SankakuSource(['{danbooru_tag}'])
            """))

        elif src == ImageSource.AUTO:
            source_imports.append("GcharAutoSource")
            source_blocks.append(textwrap.dedent(f"""\
                # ── GcharAutoSource ──────────────────────────────────────────────────
                # Automatically crawls from multiple sites using gchar character database
                # Supports game characters with multi-language name resolution
                # Install: pip install waifuc[gchar]
                auto_source = GcharAutoSource('{char}')
            """))

    # Determine source variable name for .attach() call
    if len(params.sources) == 1:
        src_var = f"{params.sources[0].value}_source" if params.sources[0] != ImageSource.AUTO else "auto_source"
    else:
        # Multiple sources — use UnionDataSource
        src_vars = []
        for src in params.sources:
            if src == ImageSource.AUTO:
                src_vars.append("auto_source")
            else:
                src_vars.append(f"{src.value}_source")
        union_vars = ", ".join(src_vars)
        source_blocks.append(textwrap.dedent(f"""\
            # ── Merge all sources ────────────────────────────────────────────────
            from waifuc.source import UnionDataSource
            source = UnionDataSource([{union_vars}])
        """))
        src_var = "source"

    if len(params.sources) == 1:
        src_var_name = src_var
    else:
        src_var_name = "source"

    # Max images slice
    max_slice = f"[:{params.max_images}]" if params.max_images else ""

    # Build the full script
    all_source_imports = ", ".join(set(source_imports))
    sources_block = "\n".join(source_blocks)

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        """
        f'"""'
        f"""
        waifuc dataset collection script for: {char}
        Model format: {params.model_format.value.upper()} ({crop_note})
        Sources: {", ".join(s.value for s in params.sources)}
        Content rating: {params.content_rating.value}
        Generated by deepghs-mcp

        Install:
            pip install git+https://github.com/deepghs/waifuc.git@main#egg=waifuc
            pip install git+https://github.com/deepghs/waifuc.git@main#egg=waifuc[gpu]  # if CUDA available

        Run:
            python crawl_{char_safe}.py
        """
        f'"""'
        f"""

        from waifuc.action import (
            NoMonochromeAction,
            FilterSimilarAction,
            TaggingAction,
            PersonSplitAction,
            FaceCountAction,
            CCIPAction,
            ModeConvertAction,
            ClassFilterAction,
            AlignMinSizeAction,
            PaddingAlignAction,
            {rating_import}
        )
        from waifuc.export import TextualInversionExporter
        from waifuc.source import {all_source_imports}

        if __name__ == '__main__':

        {textwrap.indent(sources_block, '    ')}
            # ─────────────────────────────────────────────────────────────────────
            # Pipeline: each Action processes images in sequence.
            # Actions that filter images return nothing for rejected images.
            # Actions that transform images return modified images.
            # ─────────────────────────────────────────────────────────────────────
            {src_var_name}{max_slice}.attach(

                # STEP 1 — Standardize color mode
                # Convert all images to RGB with white background (removes transparency).
                # 'white' background is standard for LoRA training.
                ModeConvertAction('RGB', 'white'),

                # STEP 2 — Pre-filter: remove bad image types
                # Drops monochrome, greyscale, and sketch images — these degrade LoRA
                # quality by teaching the model the wrong style (line art, not colored).
                NoMonochromeAction(),

                # Drops manga panels and 3D renders. Keeps only 'illustration' and
                # 'bangumi' (anime screenshot) type images.
                ClassFilterAction(['illustration', 'bangumi']),

        {textwrap.indent(rating_filter_code, '        ')}

                # STEP 3 — Deduplication
                # Drops near-identical images (like minor variation sets / 差分).
                # 'all' mode compares against entire seen set, not just recent images.
                FilterSimilarAction('all'),

                # STEP 4 — Person isolation
                # Drop images with no face detected (backgrounds, objects, etc.)
                FaceCountAction(1),

                # Split images containing multiple characters into individual crops.
                # Each crop becomes its own image for the next steps.
                PersonSplitAction(),

                # After splitting, re-filter for exactly 1 face per crop.
                FaceCountAction(1),

                # STEP 5 — CCIP Identity Filter (the magic step)
                # CCIP (Contrastive Character Image Pretraining) uses AI to verify
                # that the character in each image is actually {char}.
                # It runs a state machine: first collects reference images to build
                # a feature profile, then filters out images that don't match.
                # This achieves near-manual-selection quality automatically.
                # Without this, ~20-40% of your dataset may be wrong characters.
                CCIPAction(),

                # STEP 6 — Tagging
                # Runs WD14 tagger to generate caption tags for every image.
                # These become the .txt sidecar files used during training.
                # Tags are written to the export directory automatically.
                TaggingAction(),

                # STEP 7 — Resize and crop for training
                # Ensure minimum side is at least {min_size}px (upscales if needed).
                AlignMinSizeAction({min_size}),

                # Pad to square at {pad_size}px (adds white padding if needed).
                # This is the standard input size for {params.model_format.value.upper()} training.
                PaddingAlignAction({pad_size}),

            ).export(
                # TextualInversionExporter creates the folder structure expected by
                # kohya_ss, EveryDream, and other standard LoRA trainers:
                #   {output_dir}/{char_safe}/
                #     img/
                #       <n>_{char_safe}/     <- n = repeat count per image
                #         image_001.png
                #         image_001.txt      <- WD14 tags
                #     log/
                #     model/
                TextualInversionExporter('{output_dir}', name='{char_safe}')
            )

        print(f"Done! Dataset saved to: {output_dir}/{char_safe}/img/")
        print("Next steps:")
        print("  1. Review images in the output folder")
        print("  2. Remove any remaining bad images manually")
        print("  3. Edit .txt tag files to add your trigger word")
        print("  4. Train with kohya_ss or your preferred trainer")
    """)

    lines = [
        f"## Generated waifuc Script — `{char}`",
        f"",
        f"**Format:** {params.model_format.value.upper()} | **Sources:** {', '.join(s.value for s in params.sources)} | **Rating:** {params.content_rating.value}",
        f"**Output:** `{output_dir}/{char_safe}/`",
        f"",
        f"```python",
        script,
        f"```",
        f"",
        f"### Installation",
        f"```bash",
        f"# CPU only",
        f"pip install git+https://github.com/deepghs/waifuc.git@main#egg=waifuc",
        f"",
        f"# With GPU acceleration (recommended — CCIP and tagging are much faster)",
        f"pip install git+https://github.com/deepghs/waifuc.git@main#egg=waifuc[gpu]",
        f"```",
        f"",
        f"### Key Notes",
        f"- **CCIPAction** is the most important step — it uses AI to verify each image actually shows `{char}`",
        f"- **TaggingAction** runs WD14 tagger automatically — no manual tagging needed",
        f"- Results will be in `{output_dir}/{char_safe}/img/<n>_{char_safe}/` — compatible with kohya_ss",
        f"- First run will download model weights for CCIP + WD14 tagger (~500MB total)",
    ]

    return "\n".join(lines)


@mcp.tool(
    name="deepghs_generate_cheesechaser_script",
    annotations={
        "title": "Generate cheesechaser Download Script for DeepGHS Datasets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def deepghs_generate_cheesechaser_script(params: GenerateCheesechaserScriptInput) -> str:
    """Generate a cheesechaser Python script to download images from an indexed DeepGHS dataset.

    cheesechaser is DeepGHS's tool for selectively downloading images from HuggingFace
    datasets that are stored as indexed tar archives. Instead of downloading entire
    multi-GB tar files, you provide a list of post IDs and it extracts only those images.

    This is the most efficient way to get specific images from datasets like:
      - deepghs/danbooru2024 (~8M images, hundreds of GB total)
      - deepghs/gelbooru-webp-4Mpixel (~millions of images)
      - deepghs/sankaku_full (~millions of images)

    Args:
        params (GenerateCheesechaserScriptInput):
            - repo_id (str): HF dataset repo ID (e.g. 'deepghs/danbooru2024')
            - output_dir (str): Local directory to save downloaded images
            - post_ids (Optional[list[int]]): Specific post IDs to download
            - max_workers (int): Parallel download threads (1–16, default: 4)

    Returns:
        str: Complete cheesechaser Python script with inline comments, plus
             guidance on how to find post IDs from Danbooru/Gelbooru search results.
    """
    ids_str = repr(params.post_ids) if params.post_ids else "post_ids  # replace with your list"
    ids_example = repr(params.post_ids[:5]) if params.post_ids and len(params.post_ids) > 5 else ids_str

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        """
        f'"""'
        f"""
        cheesechaser download script for: {params.repo_id}
        Output: {params.output_dir}
        Workers: {params.max_workers}
        Generated by deepghs-mcp

        Install:
            pip install cheesechaser

        cheesechaser downloads images from HuggingFace indexed tar datasets
        without needing to download the entire archive.
        """
        f'"""'
        f"""

        from cheesechaser.datapool import SimpleDataPool

        # Initialize the data pool for this dataset
        pool = SimpleDataPool('{params.repo_id}')

        # ─────────────────────────────────────────────────────────────────────────
        # Option A: Download specific post IDs
        # ─────────────────────────────────────────────────────────────────────────
        # Get post IDs from a Danbooru/Gelbooru search, or from the dataset's
        # Parquet metadata files (which contain all IDs with their tags).
        post_ids = {ids_str}

        pool.batch_download_to_directory(
            resource_ids=post_ids,
            local_directory='{params.output_dir}',
            max_workers={params.max_workers},
        )

        # ─────────────────────────────────────────────────────────────────────────
        # Option B: Download all available images (WARNING: may be hundreds of GB)
        # ─────────────────────────────────────────────────────────────────────────
        # Uncomment to download everything:
        # pool.batch_download_to_directory(
        #     resource_ids=None,  # None = all
        #     local_directory='{params.output_dir}',
        #     max_workers={params.max_workers},
        # )

        # ─────────────────────────────────────────────────────────────────────────
        # Option C: Get IDs from the dataset's Parquet index, filtered by tags
        # ─────────────────────────────────────────────────────────────────────────
        # import pandas as pd
        # df = pd.read_parquet('hf://datasets/{params.repo_id}/metadata.parquet')
        # # Filter by tag — e.g. only images tagged 'hatsune_miku'
        # filtered = df[df['tags'].str.contains('hatsune_miku', na=False)]
        # tag_post_ids = filtered['id'].tolist()
        # pool.batch_download_to_directory(
        #     resource_ids=tag_post_ids,
        #     local_directory='{params.output_dir}/hatsune_miku',
        #     max_workers={params.max_workers},
        # )

        print(f"Download complete: {params.output_dir}")
    """)

    lines = [
        f"## Generated cheesechaser Script — `{params.repo_id}`",
        f"",
        f"**Dataset:** [{params.repo_id}](https://huggingface.co/datasets/{params.repo_id})",
        f"**Output:** `{params.output_dir}` | **Workers:** {params.max_workers}",
        f"",
        f"```python",
        script,
        f"```",
        f"",
        f"### Installation",
        f"```bash",
        f"pip install cheesechaser",
        f"```",
        f"",
        f"### How to Find Post IDs",
        f"1. **From Danbooru search**: `https://danbooru.donmai.us/posts?tags=hatsune_miku` — IDs are in the URL of each post",
        f"2. **From the dataset Parquet**: Use Option C above to filter by tag from the metadata file",
        f"3. **From deepghs_search_tags**: Cross-reference tags, then filter the Parquet by those tag names",
        f"4. **Existing character dataset**: Use `deepghs_find_character_dataset` to find pre-built ID lists",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    """Entry point for PyPI script installation (deepghs-mcp command)."""
    mcp.run()


if __name__ == "__main__":
    main()
