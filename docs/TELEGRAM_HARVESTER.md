# Telegram Harvester (`telegram_downloader.py`)

## High-Level Purpose
This module acts as the Standalone Headless Harvester for the Video Intelligence OS. It connects to a private Telegram channel via Pyrogram, resolves MTProto cache amnesia autonomously, extracts structured metadata from captions using regex, and stages raw `.mp4` files onto the local disk for GPU processing.

## Architecture & Logic Breakdown

### 1. Initialization & Concurrency
* **`nest_asyncio.apply()`**: Allows Pyrogram's event loop to run concurrently within Kaggle's existing IPython event loop.
* **`custom_print()`**: Wraps standard print to append timestamps and force `flush=True`, bypassing Kaggle's stdout buffering for real-time log monitoring.

### 2. Database Layer (`init_db`)
* Establishes `vios_master.db` using SQLite.
* Enforces `PRAGMA journal_mode=WAL;` to allow simultaneous read/write operations (critical when the Downloader writes metadata while the GPU orchestrator reads videos).
* Uses a flattened schema for maximum read performance.

### 3. The Amnesia Protocol
* **The Bug**: MTProto clients often lose the `access_hash` for private channels across cold reboots.
* **The Fix**: The script attempts to send a silent ping. Upon `PeerIdInvalid` failure, it utilizes the HTTP REST API (`api.telegram.org`) to force a message into the channel. The Pyrogram client listens for this specific message via an `@app_client.on_message()` decorator, catching the event and securely caching the required hash in memory.

### 4. Flash Scan & Metadata Ingestion
* Drops a transient message to acquire the absolute `latest_id` of the channel.
* Compares `latest_id` against the SQLite database to identify missing `msg_id` gaps.
* Fetches missing messages in batches of 200 to prevent rate limiting.
* Applies regex (`extract_text`, `extract_num`) to parse Categories, Creators, Likes, and Captions, inserting them as `Metadata_Only`.

### 5. Media Harvesting
* Queries SQLite for `Metadata_Only` rows.
* Respects `CATEGORY_QUEUE` to allow dynamic prioritization of specific topics.
* Streams the `.mp4` payload to the `archive/` directory and updates the database state to `Harvested`, marking it ready for Stream B (GPU 0).
