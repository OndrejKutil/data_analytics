# Concurrent Scraping and Incremental Saving

This document explains how concurrent scraping works in this project, why memory usage can spike, and how to save incrementally to avoid holding all results in memory.

## Current Flow

- Scraper: `DetailScraper.scrape_multiple()` runs requests in batches (controlled by `BATCH_SIZE`).
- Concurrency: Each batch schedules multiple `scrape()` tasks via `asyncio.gather`.
- Aggregation: Results from each batch are appended to a growing in-memory list `results`.
- Saving: If `SAVE_FILES` and `save_path` are set, `_save_results(results, save_path)` converts the entire list to a DataFrame and writes CSV, overwriting the file each time.

## Why Memory Spikes

- The `results` list grows with every batch, so all scraped listings live in memory until the end.
- Some fields (e.g., `images`, `videos`, sets of utilities) are lists converted to JSON strings for CSV; keeping full history increases object count.
- Converting a large list of dicts to a DataFrame repeatedly is expensive.
- Observed: VS Code used ~9 GB RAM during scraping, which aligns with holding many pages of detailed results.

## Better Approach: Incremental Append (Streaming)

Instead of keeping all results in memory and overwriting, write only each batch’s results to disk and release memory immediately.

### Recommended changes

- Convert only the batch’s results to a DataFrame.
- Append to CSV using `mode='a'` and write the header only once.
- Keep an `exists = save_path.exists()` check to decide whether to write the header.
- Do not pass the full `results` into the saving function—pass just `batch_results`.

```python
# In scrape_multiple(...)
# After: batch_results = await asyncio.gather(*tasks)
self._save_batch(batch_results, save_path)

# New method
def _save_batch(self, batch_results: list[dict[str, Any]], save_path: Path) -> None:
    if not batch_results:
        return

    df = pd.DataFrame(batch_results)

    list_columns = [
        'images', 'videos', 'seller_phones', 'heating_types', 'heating_sources',
        'heating_elements', 'water_heat_sources', 'water_types', 'sewage_types',
        'gas_types', 'electricity_types', 'telecommunication_types',
        'internet_connection_types', 'transport_types', 'road_types'
    ]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)

    write_header = not save_path.exists()
    df.to_csv(save_path, index=False, encoding='utf-8', mode='a', header=write_header)
```

### Pros

- Memory remains bounded by `BATCH_SIZE`.
- Frequent incremental checkpoints reduce data loss on failures.

### Cons / Caveats

- CSV schema must remain stable across batches; if fields appear later, they won’t have headers. Prefer defining a stable column set early or write a schema header first.
- Appends are not atomic; in case of interruption during write, last batch may be partially written. For stronger guarantees, write to a temp file and replace.

## Alternative Storage Options

- Parquet: `df.to_parquet(path, index=False)` with partitioning; smaller size, faster IO, preserves types and lists via nesting, but needs Arrow/Parquet libs.
- SQLite/DuckDB: Append rows transactionally; robust, schema-safe, and supports later SQL transforms.
- NDJSON: Write one JSON per line; excellent for streaming and later ingestion.

## Concurrency Patterns

- Keep `BATCH_SIZE` tuned (e.g., 10–30) to avoid server overload and local resource spikes.
- Optional: writer queue pattern
  - Have scraping tasks `put_nowait()` results into an `asyncio.Queue`.
  - A dedicated writer coroutine consumes the queue and flushes to disk in small chunks.
  - This decouples network concurrency from IO and further limits memory.

## Failure Recovery & Resuming

- With appends, you can resume runs and deduplicate by a stable key (e.g., `listing_id`).
- Consider maintaining a lightweight index (set of seen `listing_id`s) to skip already-scraped items when resuming.

## Practical Tips

- If you don’t need heavy arrays (e.g., image metadata), drop them before writing.
- If bandwidth is the bottleneck, increasing `BATCH_SIZE` helps; if memory/CPU is stressed, reduce it.
- Periodically `gc.collect()` isn’t a fix for growth—streaming writes are.
