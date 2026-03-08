# deepghs-mcp

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![MCP](https://img.shields.io/badge/MCP-compatible-green)


A Python [MCP](https://modelcontextprotocol.io/) server for the [DeepGHS](https://huggingface.co/deepghs) anime AI ecosystem. Connect it to any MCP-compatible client (Claude Desktop, Cursor, etc.) to browse datasets, discover pre-built character training sets, look up tags across 18 platforms, and generate complete data pipeline scripts — all directly from your AI assistant.

---

## ✨ Features

### 📦 Dataset & Model Discovery

- **Browse all DeepGHS datasets** — Danbooru2024 (8M+ images), Sankaku, Gelbooru, Zerochan, BangumiBase, and more
- **Full file trees** — see exactly which tar/parquet files a dataset contains and how large they are before downloading anything
- **Model catalog** — find the right model for your task: CCIP, WD Tagger Enhanced, aesthetic scorer, face/head detector, anime classifier, and more
- **Live demos** — browse DeepGHS Spaces (interactive web apps) for testing models without code

### 🏷️ Cross-Platform Tag Intelligence

- **site_tags lookup** — 2.5M+ tags unified across 18 platforms in one query
- **Tag format translation** — Danbooru uses `hatsune_miku`, Zerochan uses `Hatsune Miku`, Pixiv uses `初音ミク` — this tool maps them all together
- **Ready-to-use Parquet queries** — get copy-paste code to filter the tag database programmatically

### 🎯 Character Dataset Finder

- **Pre-built LoRA datasets** — search both `deepghs` and `CyberHarem` namespaces for existing character image collections
- **Ready-to-run download commands** — get the exact `cheesechaser` command to pull what you need
- **Smart fallback** — if no pre-built dataset exists, the tool hands off directly to the waifuc script generator

### 🤖 Training Pipeline Code Generation

- **waifuc scripts** — generate complete, annotated Python data collection pipelines for any character from any source (Danbooru, Pixiv, Gelbooru, Zerochan, Sankaku, or Auto)
- **cheesechaser scripts** — generate targeted download scripts to pull specific post IDs from indexed multi-TB datasets without downloading the whole archive
- **Format-aware** — crop sizes, bucket ranges, and export formats automatically adjusted for SD 1.5, SDXL, or Flux

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- `git`

### Quick Start

1. **Clone the repository:**

```bash
git clone https://github.com/citronlegacy/deepghs-mcp.git
cd deepghs-mcp
```

2. **Run the installer:**

```bash
chmod +x install.sh && ./install.sh
# or without chmod:
bash install.sh
```

3. **Or install manually:**

```bash
pip install -r requirements.txt
```

---

## 🔑 Authentication

`HF_TOKEN` is optional for public datasets but strongly recommended — it raises HuggingFace's API rate limit and is required for any gated or private repositories.

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is sufficient).

Without it, the server still works for all public DeepGHS datasets.

---

## ▶️ Running the Server

```bash
python deepghs_mcp.py
# or via the venv created by install.sh:
.venv/bin/python deepghs_mcp.py
```

---

## ⚙️ Configuration

### Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "deepghs": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/deepghs_mcp.py"],
      "env": {
        "HF_TOKEN": "hf_your_token_here"
      }
    }
  }
}
```

### Other MCP Clients

- **Command**: `/absolute/path/to/.venv/bin/python`
- **Args**: `/absolute/path/to/deepghs_mcp.py`
- **Transport**: stdio

---

## 💡 Usage Examples

### Browse available datasets

> "What anime datasets does DeepGHS have on HuggingFace?"

The assistant calls `deepghs_list_datasets` and returns all datasets sorted by download count — Danbooru2024, Sankaku, Gelbooru WebP, BangumiBase, site_tags, and more — with links and update dates.

---

### Check dataset contents before downloading

> "What files are in deepghs/danbooru2024? How big is it?"

The assistant calls `deepghs_get_repo_info` and returns the full file tree — every `.tar` and `.parquet` file with individual and total sizes — so you know exactly what you're committing to before you download.

---

### Find a pre-built character dataset

> "Is there already a dataset for Rem from Re:Zero I can use for LoRA training?"

The assistant calls `deepghs_find_character_dataset`, searches both `deepghs` and `CyberHarem` namespaces, and returns any matches with download counts and a one-liner download command.

**Example response:**
```
## Character Dataset Search: Rem

Found 2 dataset(s):

### CyberHarem/rem_rezero
Downloads: 4.2K | Likes: 31 | Updated: 2024-08-12

Download with cheesechaser:
  from cheesechaser.datapool import SimpleDataPool
  pool = SimpleDataPool('CyberHarem/rem_rezero')
  pool.batch_download_to_directory('./dataset', max_workers=4)
```

---

### Look up a tag across all platforms

> "What is the correct tag for 'Hatsune Miku' on Danbooru, Zerochan, and Pixiv?"

The assistant calls `deepghs_search_tags` and returns the format for every platform, plus a pre-filtered link to the site_tags dataset viewer and Parquet query code.

**Example response:**
```
Tag Lookup: Hatsune Miku

  Danbooru:  hatsune_miku
  Gelbooru:  hatsune_miku
  Sankaku:   character:hatsune_miku
  Zerochan:  Hatsune Miku
  Pixiv:     初音ミク (Japanese preferred)
  Yande.re:  hatsune_miku
  Wallhaven: hatsune miku
```

This directly solves the MultiBoru tag normalization problem.

---

### Generate a full data collection pipeline

> "Generate a waifuc script to build a LoRA dataset for Surtr from Arknights, from Danbooru and Pixiv, for SDXL, safe-only."

The assistant calls `deepghs_generate_waifuc_script` and returns a complete annotated Python script. Here is what that script does when run:

```
1.  Crawls Danbooru (surtr_(arknights)) + Pixiv (スルト アークナイツ)
2.  Converts all images to RGB with white background
3.  Drops monochrome, sketch, manga panels, and 3D renders
4.  Filters to safe rating only
5.  Deduplicates near-identical images (差分 / variants)
6.  Drops images with no detected face
7.  Splits group images — each character becomes its own crop
8.  CCIP identity filter — AI verifies every image actually shows Surtr
9.  WD14 auto-tagging — writes .txt caption files automatically
10. Resizes and pads to 1024×1024 (SDXL standard)
11. Exports in kohya_ss-compatible folder structure
```

No manual curation. No manual tagging. Drop the output folder straight into your trainer.

---

### Generate a targeted download script

> "Give me a cheesechaser script to download post IDs 1234, 5678, 9012 from deepghs/danbooru2024."

The assistant calls `deepghs_generate_cheesechaser_script` and returns a complete Python script — with options for downloading by post ID list, downloading everything, or filtering from the Parquet index by tag.

---

### Find the right model for a task

> "Does DeepGHS have a model for scoring image aesthetics?"

The assistant calls `deepghs_list_models` with `search: "aesthetic"` and returns matching models with pipeline task, download counts, and direct HuggingFace links.

---

### Try a model in the browser

> "Is there a live demo for the DeepGHS face detection model?"

The assistant calls `deepghs_list_spaces` with `search: "detection"` and returns matching Spaces with direct links to the interactive demos.

---

## 🛠️ Available Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `deepghs_list_datasets` | Browse all DeepGHS datasets with search, sort, and pagination | `search`, `sort`, `limit`, `offset` |
| `deepghs_list_models` | Browse all DeepGHS models | `search`, `sort`, `limit` |
| `deepghs_list_spaces` | Browse all DeepGHS live demo Spaces | `search`, `limit` |
| `deepghs_get_repo_info` | Full file tree + metadata for any dataset/model/space | `repo_id`, `repo_type` |
| `deepghs_search_tags` | Cross-platform tag lookup across 18 platforms via site_tags | `tag` |
| `deepghs_find_character_dataset` | Find pre-built LoRA training datasets for a character | `character_name` |
| `deepghs_generate_waifuc_script` | Generate complete data collection + cleaning pipeline script | `character_name`, `sources`, `model_format`, `content_rating` |
| `deepghs_generate_cheesechaser_script` | Generate targeted dataset download script | `repo_id`, `post_ids`, `output_dir` |

---

## 📖 Tools Reference

### `deepghs_list_datasets`

Lists all public datasets from DeepGHS, sortable and filterable by keyword.

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `search` | string | ❌ | — | Keyword filter, e.g. `danbooru`, `character`, `face` |
| `sort` | string | ❌ | `downloads` | `downloads`, `likes`, `createdAt`, `lastModified` |
| `limit` | integer | ❌ | `20` | Results per page (max 100) |
| `offset` | integer | ❌ | `0` | Pagination offset |
| `response_format` | string | ❌ | `markdown` | `markdown` or `json` |

---

### `deepghs_list_models`

Lists all public models — CCIP, WD Tagger Enhanced, aesthetic scorer, face/head/person detectors, anime classifier, furry detector, NSFW censor, style era classifier, and more.

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `search` | string | ❌ | — | Keyword filter, e.g. `ccip`, `tagger`, `aesthetic`, `face` |
| `sort` | string | ❌ | `downloads` | `downloads`, `likes`, `createdAt`, `lastModified` |
| `limit` | integer | ❌ | `20` | Results per page (max 100) |
| `offset` | integer | ❌ | `0` | Pagination offset |
| `response_format` | string | ❌ | `markdown` | `markdown` or `json` |

---

### `deepghs_list_spaces`

Lists all public Spaces — live demos for face detection, head detection, CCIP character similarity, WD tagger, aesthetic scorer, reverse image search, Danbooru character lookup, and more.

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `search` | string | ❌ | — | Keyword filter, e.g. `detection`, `tagger`, `search` |
| `limit` | integer | ❌ | `20` | Results per page (max 100) |
| `response_format` | string | ❌ | `markdown` | `markdown` or `json` |

---

### `deepghs_get_repo_info`

Get full metadata for any dataset, model, or space — including the complete file tree with individual file sizes. Essential before downloading a multi-TB dataset.

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `repo_id` | string | ✅ | — | Full HF repo ID, e.g. `deepghs/danbooru2024` |
| `repo_type` | string | ❌ | `dataset` | `dataset`, `model`, or `space` |
| `response_format` | string | ❌ | `markdown` | `markdown` or `json` |

> **Tip:** Datasets containing `.tar` files automatically get a ready-to-copy cheesechaser snippet appended.

---

### `deepghs_search_tags`

Look up any tag across 18 platforms using the `deepghs/site_tags` dataset. Returns per-platform format guidance, a pre-filtered dataset viewer link, and Parquet query code.

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `tag` | string | ✅ | — | Tag in any format or language, e.g. `hatsune_miku`, `Hatsune Miku`, `初音ミク` |
| `response_format` | string | ❌ | `markdown` | `markdown` or `json` |

> **LLM Tip:** Call this before `deepghs_generate_waifuc_script` to confirm the correct Danbooru/Pixiv tag format for a character.

---

### `deepghs_find_character_dataset`

Searches `deepghs` and `CyberHarem` on HuggingFace for pre-built character datasets. CyberHarem datasets are built with the full automated pipeline: crawl → CCIP filter → WD14 tag → upload.

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `character_name` | string | ✅ | — | Character name, e.g. `Rem`, `Hatsune Miku`, `surtr arknights` |
| `response_format` | string | ❌ | `markdown` | `markdown` or `json` |

---

### `deepghs_generate_waifuc_script`

Generates a complete, annotated Python pipeline script using [waifuc](https://github.com/deepghs/waifuc).

**Pipeline actions included (in order):**

| Action | What it does | Why it matters for LoRA |
|---|---|---|
| `ModeConvertAction` | Convert to RGB, white background | Standardizes input format |
| `NoMonochromeAction` | Drop greyscale/sketch images | Prevents style contamination |
| `ClassFilterAction` | Keep illustration/anime only | Drops manga panels and 3D |
| `RatingFilterAction` | Filter by content rating | Keep dataset SFW if needed |
| `FilterSimilarAction` | Deduplicate similar images | Prevents overfitting to variants |
| `FaceCountAction` | Require exactly 1 face | Removes group shots and objects |
| `PersonSplitAction` | Crop each character from group images | Maximizes usable data |
| `CCIPAction` | AI identity verification | Removes wrong characters (the most important step) |
| `TaggingAction` | WD14 auto-tagging | Generates `.txt` captions automatically |
| `AlignMinSizeAction` | Resize to minimum resolution | Ensures quality floor |
| `PaddingAlignAction` | Pad to square | Standard training resolution |

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `character_name` | string | ✅ | — | Display name, e.g. `Rem`, `Surtr` |
| `danbooru_tag` | string | ❌ | auto-guessed | Danbooru tag, e.g. `rem_(re:zero)` |
| `pixiv_query` | string | ❌ | — | Pixiv search query, Japanese preferred. Required if `pixiv` in sources |
| `sources` | list | ❌ | `["danbooru"]` | `danbooru`, `pixiv`, `gelbooru`, `zerochan`, `sankaku`, `auto` |
| `model_format` | string | ❌ | `sd1.5` | `sd1.5` (512px), `sdxl` (1024px), or `flux` (1024px) |
| `content_rating` | string | ❌ | `safe` | `safe`, `safe_r15`, or `all` |
| `output_dir` | string | ❌ | `./dataset_output` | Output directory |
| `max_images` | integer | ❌ | no limit | Cap total images collected |
| `pixiv_token` | string | ❌ | — | Pixiv refresh token (required for Pixiv source) |

**Source tag formats:**

| Source | Tag Format | Notes |
|---|---|---|
| `danbooru` | `rem_(re:zero)` | snake_case with series in parens |
| `gelbooru` | `rem_(re:zero)` | same as Danbooru |
| `pixiv` | `レム` / `rem re:zero` | Japanese preferred for better results |
| `zerochan` | `Rem` | Title Case, strict mode enabled |
| `sankaku` | `rem_(re:zero)` | snake_case |
| `auto` | character name | uses gchar database — best for game characters |

---

### `deepghs_generate_cheesechaser_script`

Generates a Python download script using [cheesechaser](https://github.com/deepghs/cheesechaser) to pull specific images from indexed tar datasets without downloading entire archives.

**Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `repo_id` | string | ✅ | — | HF dataset, e.g. `deepghs/danbooru2024` |
| `output_dir` | string | ❌ | `./downloads` | Local download directory |
| `post_ids` | list[int] | ❌ | — | Specific post IDs. If omitted, downloads all (can be very large) |
| `max_workers` | integer | ❌ | `4` | Parallel download threads (1–16) |

---

## 🗂️ Key DeepGHS Datasets

| Repo ID | Description | Use Case |
|---|---|---|
| `deepghs/danbooru2024` | Full Danbooru archive, 8M+ images | Bulk downloads, data mining |
| `deepghs/danbooru2024-webp-4Mpixel` | Compressed WebP version | Faster downloads |
| `deepghs/sankaku_full` | Full Sankaku Channel dataset | Alternative tag ecosystem |
| `deepghs/gelbooru-webp-4Mpixel` | Gelbooru compressed | Western fanart coverage |
| `deepghs/site_tags` | 2.5M+ tags, 18 platforms | Tag normalization |
| `deepghs/anime_face_detection` | YOLO face detection labels | Train detection models |
| `deepghs/bangumibase` | Character frames from anime | Character dataset bootstrapping |

---

## 🔄 Recommended Workflows

### Workflow A: Use a pre-built dataset

```
1. deepghs_find_character_dataset   → check if dataset exists
2. deepghs_get_repo_info            → inspect file sizes
3. deepghs_generate_cheesechaser_script → get download command
4. Run the script → train
```

### Workflow B: Build a new dataset from scratch

```
1. deepghs_search_tags              → find the correct Danbooru tag
2. deepghs_generate_waifuc_script   → generate full pipeline script
3. Run the script                   → crawl, filter, tag, crop
4. Review output                    → remove any remaining noise
5. Train with kohya_ss
```

### Workflow C: Mine specific images from a large dataset

```
1. deepghs_get_repo_info            → inspect dataset Parquet structure
2. deepghs_search_tags              → confirm tag names
3. deepghs_generate_cheesechaser_script → generate Parquet-filter script
4. Run the script                   → downloads only matching images
```

---

## 🤖 Notes for LLMs

- **Check pre-built first**: Always call `deepghs_find_character_dataset` before generating a waifuc script. CyberHarem has hundreds of ready-to-use LoRA datasets.
- **Danbooru tag format**: `character_(series)` with underscores — `rem_(re:zero)`, not `Rem (Re:Zero)`. Use `deepghs_search_tags` to confirm.
- **File sizes**: Datasets like `danbooru2024` are multi-TB. Always check `deepghs_get_repo_info` before recommending a full download. Use cheesechaser with post IDs for targeted access.
- **CCIP is essential**: It's the most important pipeline step — without it, 20–40% of a character dataset will be wrong-character noise. Always include it in waifuc scripts.
- **Pixiv source**: Requires a `pixiv_token`. If the user hasn't set one up, suggest Danbooru + Gelbooru instead.
- **Model format crop sizes**: SD1.5 = 512×512, SDXL/Flux = 1024×1024. This controls `AlignMinSizeAction` and `PaddingAlignAction` in generated scripts.

---

## ⚠️ Known Limitations

- **HuggingFace rate limits**: Without `HF_TOKEN`, the Hub API may throttle requests on heavy usage.
- **Gated datasets**: Some datasets require explicit approval on HuggingFace before downloading. The server returns a clear error with guidance.
- **CyberHarem search**: Niche characters may need manual browsing at [huggingface.co/CyberHarem](https://huggingface.co/CyberHarem).
- **waifuc runtime**: The generated scripts require `waifuc` installed separately (not in this server's deps). First run downloads ~500MB of CCIP + WD14 model weights.

---

## 🐛 Troubleshooting

**Server won't start:**
- Ensure Python 3.10+: `python --version`
- Re-run the installer: `bash install.sh`

**Rate limit / 429 errors:**
- Set `HF_TOKEN` in your MCP env config
- Get a free token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**403 / Forbidden on a dataset:**
- The dataset is gated — visit the dataset page on HuggingFace and click "Request Access"
- Ensure the `HF_TOKEN` is from an account that has been granted access

**Character dataset not found:**
- Try alternate spellings: `"Rem"`, `"rem_(re:zero)"`, `"rem re:zero"`
- Browse manually: [huggingface.co/CyberHarem](https://huggingface.co/CyberHarem)
- Generate from scratch with `deepghs_generate_waifuc_script`

**waifuc script fails:**
- Install waifuc: `pip install git+https://github.com/deepghs/waifuc.git`
- GPU support: `pip install "waifuc[gpu]"` (much faster CCIP + tagging)
- First run downloads ~500MB of model weights — this is expected

---

## 🤝 Contributing

Pull requests are welcome! If a tool is returning incorrect data, a script template is outdated, or a new DeepGHS dataset or model should be highlighted, please open an issue or PR.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- 🤗 [DeepGHS on HuggingFace](https://huggingface.co/deepghs)
- 🧰 [waifuc — data pipeline](https://github.com/deepghs/waifuc)
- 🧀 [cheesechaser — targeted HF downloader](https://github.com/deepghs/cheesechaser)
- 🖼️ [imgutils — image processing](https://github.com/deepghs/imgutils)
- 🤖 [cyberharem — automated LoRA pipeline](https://github.com/deepghs/cyberharem)
- 🔧 [MCP documentation](https://modelcontextprotocol.io/)
- 🐛 [Bug Reports](https://github.com/citronlegacy/deepghs-mcp/issues)
- 💡 [Feature Requests](https://github.com/citronlegacy/deepghs-mcp/discussions)

### Related MCP Servers

- [gelbooru-mcp](https://github.com/citronlegacy/gelbooru-mcp) — Search Gelbooru, generate SD prompts from character tag data
- zerochan-mcp — Browse Zerochan's high-quality anime image board

<!-- mcp-name: io.github.citronlegacy/deepghs-mcp -->
