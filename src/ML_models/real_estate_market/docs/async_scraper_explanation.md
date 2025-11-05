# Asynchronous Web Scraper Implementation

## Overview

This document explains the transformation of the web scraper from synchronous to asynchronous execution, resulting in an **8-10x performance improvement**.

---

## Table of Contents

1. [What Changed](#what-changed)
2. [How Async/Await Works](#how-asyncawait-works)
3. [Performance Comparison](#performance-comparison)
4. [Code Breakdown](#code-breakdown)
5. [Key Concepts](#key-concepts)
6. [Best Practices](#best-practices)

---

## What Changed

### Dependencies

**Before:**
```python
import requests  # Synchronous HTTP library
```

**After:**
```python
import asyncio   # Python's async framework
import aiohttp   # Asynchronous HTTP library
```

### Execution Model

| Aspect | Synchronous (Before) | Asynchronous (After) |
|--------|---------------------|---------------------|
| Request handling | One at a time | Multiple concurrent |
| Waiting behavior | Blocks entire program | Switches to other tasks |
| Speed (44 listings) | ~88 seconds | ~10 seconds |
| Scalability | Linear growth | Sub-linear growth |

---

## How Async/Await Works

### Mental Model: The Restaurant Analogy

#### **Synchronous Approach (Before):**
```
Waiter serves customers ONE at a time:
🍽️ Take order from Table 1 → wait for kitchen → serve → collect payment
🍽️ Take order from Table 2 → wait for kitchen → serve → collect payment
🍽️ Take order from Table 3 → wait for kitchen → serve → collect payment

Total time: 30 minutes (10 min × 3 tables)
```

#### **Asynchronous Approach (After):**
```
Waiter serves customers CONCURRENTLY:
🍽️ Take order from Table 1 → give to kitchen → don't wait, go to Table 2
🍽️ Take order from Table 2 → give to kitchen → don't wait, go to Table 3
🍽️ Take order from Table 3 → give to kitchen
🍽️ Kitchen ready for Table 1 → serve
🍽️ Kitchen ready for Table 2 → serve
🍽️ Kitchen ready for Table 3 → serve

Total time: 12 minutes (work on all tables during wait times)
```

### Key Principle

**Web scraping is I/O-bound:**
- 90% of time spent waiting for HTTP responses
- 10% of time spent parsing HTML
- **Perfect use case for async!**

---

## Performance Comparison

### Scenario: Scraping 10 pages + 44 listing details

#### **Synchronous (Before)**

```python
# Link scraping (sequential)
for page in range(1, 11):
    fetch_page(page)  # 1 second each
# Time: 10 seconds

# Detail scraping (sequential)
for url in listing_urls:  # 44 URLs
    fetch_details(url)  # 2 seconds each
# Time: 88 seconds

Total: 98 seconds ≈ 1.6 minutes
```

**Timeline:**
```
Page 1:  [====]
Page 2:       [====]
Page 3:            [====]
...
Detail 1:                [======]
Detail 2:                       [======]
Detail 3:                              [======]
...
```

#### **Asynchronous (After)**

```python
# Link scraping (concurrent)
await asyncio.gather(*[fetch_page(p) for p in range(1, 11)])
# Time: ~2 seconds (all at once!)

# Detail scraping (batched, 10 at a time)
# Batch 1: 10 listings concurrently
# Batch 2: 10 listings concurrently
# Batch 3: 10 listings concurrently
# Batch 4: 10 listings concurrently
# Batch 5: 4 listings concurrently
# Time: ~10 seconds (5 batches × 2 seconds)

Total: 12 seconds

Speed Improvement: 8.2x faster! 🚀
```

**Timeline:**
```
Pages 1-10: [====] All at once!

Batch 1 (Details 1-10):   [======] All at once!
Batch 2 (Details 11-20):  [======] All at once!
Batch 3 (Details 21-30):  [======] All at once!
Batch 4 (Details 31-40):  [======] All at once!
Batch 5 (Details 41-44):  [======] All at once!
```

### Scalability Analysis

| Number of Listings | Synchronous | Asynchronous | Speed-up |
|-------------------|-------------|--------------|----------|
| 10 listings | 20 sec | 3 sec | 6.7x |
| 50 listings | 100 sec | 12 sec | 8.3x |
| 100 listings | 200 sec | 22 sec | 9.1x |
| 500 listings | 1000 sec (16.7 min) | 102 sec (1.7 min) | 9.8x |

**Note:** Async scales better with more listings!

---

## Code Breakdown

### 1. Base Scraper - Async fetch_page()

**Before (Synchronous):**
```python
def fetch_page(self, url: str) -> str:
    response = self.session.get(url)  # ⏸️ Blocks here
    response.raise_for_status()
    return response.text
```

**After (Asynchronous):**
```python
async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:  # 🚀 Non-blocking
        response.raise_for_status()
        return await response.text()  # Wait only when needed
```

**Changes explained:**
1. **`async def`** - Declares function as asynchronous (returns a coroutine)
2. **`session` parameter** - Shares connection pool across requests
3. **`async with`** - Asynchronous context manager
4. **`await response.text()`** - Pauses function, lets others run

### 2. Link Scraper - Concurrent page fetching

**Before (Sequential):**
```python
def scrape(self) -> list[str]:
    self.links = []
    for page in range(1, self.num_pages + 1):
        url = f"{self.pages_url}?strana={page}"
        html = self.fetch_page(url)  # ⏸️ Wait for each page
        soup = self.parse_html(html)
        links = self.parse(soup)
        self.links.extend(links)
    return self.links
```

**After (Concurrent):**
```python
async def scrape(self) -> list[str]:
    self.links = []
    
    async with aiohttp.ClientSession(headers=self.headers) as session:
        tasks = []
        # Create all tasks (don't execute yet)
        for page in range(1, self.num_pages + 1):
            url = f"{self.pages_url}?strana={page}"
            tasks.append(self._scrape_page(session, url))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Collect results
        for links in results:
            self.links.extend(links)
    
    self._clean_links()
    return self.links

async def _scrape_page(self, session: aiohttp.ClientSession, url: str) -> list[str]:
    """Helper to scrape a single page"""
    html = await self.fetch_page(session, url)
    soup = self.parse_html(html)
    return self.parse(soup)
```

**Key concepts:**
- **Task creation:** `tasks.append(...)` creates tasks without executing
- **Concurrent execution:** `asyncio.gather(*tasks)` runs all at once
- **Results:** Returns in same order as task list

### 3. Detail Scraper - Batched concurrent fetching

**Before (Sequential):**
```python
def scrape_multiple(self, listing_urls: list[str]) -> list[dict]:
    results = []
    for url in listing_urls:
        try:
            data = self.scrape(url)  # ⏸️ One at a time
            results.append(data)
        except Exception as e:
            print(f"Error: {e}")
    return results
```

**After (Batched Concurrent):**
```python
async def scrape_multiple(self, listing_urls: list[str], 
                         batch_size: int = 10) -> list[dict]:
    """
    Scrape multiple URLs with controlled concurrency
    
    Args:
        listing_urls: List of URLs to scrape
        batch_size: Max concurrent requests (default: 10)
    """
    results = []
    
    async with aiohttp.ClientSession(headers=self.headers) as session:
        # Process in batches
        for i in range(0, len(listing_urls), batch_size):
            batch = listing_urls[i:i + batch_size]
            print(f"Scraping batch {i//batch_size + 1}")
            
            # Create tasks for this batch
            tasks = [self._scrape_single_safe(session, url) 
                    for url in batch]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Polite delay between batches
            if i + batch_size < len(listing_urls):
                await asyncio.sleep(0.5)
    
    return results

async def _scrape_single_safe(self, session, url: str) -> dict:
    """Scrape single URL with error handling"""
    try:
        return await self.scrape(session, url)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {'url': url, 'error': str(e)}
```

**Why batching?**
- **Without batching:** 44 concurrent requests → might trigger rate limiting
- **With batching:** 10 concurrent requests → polite, controlled, still fast
- **Adjustable:** Increase `batch_size` for faster (but riskier) scraping

### 4. Main Execution

**Before:**
```python
if __name__ == "__main__":
    scraper = LinkScraper(num_pages=10)
    links = scraper.scrape()
    
    detail_scraper = DetailScraper()
    details = detail_scraper.scrape_multiple(links)
```

**After:**
```python
async def main():
    """Main async function"""
    scraper = LinkScraper(num_pages=10)
    links = await scraper.scrape()  # Must await
    
    detail_scraper = DetailScraper()
    details = await detail_scraper.scrape_multiple(links, batch_size=10)
    
    # Save results
    path = Path(__file__).parent / 'listings.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(main())  # Entry point for async code
```

**Critical points:**
- **`asyncio.run(main())`** - Starts the event loop (only call once)
- **`await`** - Required for all async function calls
- **Regular I/O** (file writing) still works normally

---

## Key Concepts

### 1. Coroutines (`async def`)

```python
async def my_function():
    return "Hello"

# This is a coroutine object (not the result!)
coro = my_function()

# To get the result, you must await it
result = await my_function()  # "Hello"
```

**Rules:**
- `async def` creates a coroutine function
- Can only be awaited inside another async function
- Top-level entry point: `asyncio.run()`

### 2. Awaiting (`await`)

```python
async def fetch_data():
    # Pause here, let other tasks run
    data = await slow_network_call()
    
    # Resume here when data arrives
    return data
```

**What happens:**
1. Execution reaches `await`
2. Function pauses (but doesn't block!)
3. Event loop switches to another task
4. When result ready, function resumes

### 3. Concurrent Execution (`asyncio.gather`)

```python
# Run multiple coroutines concurrently
results = await asyncio.gather(
    fetch_page(1),
    fetch_page(2),
    fetch_page(3)
)
# results = [result1, result2, result3]
```

**Variants:**
```python
# Create tasks from list
tasks = [fetch_page(i) for i in range(10)]
results = await asyncio.gather(*tasks)

# With error handling (return exceptions instead of raising)
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 4. Context Managers (`async with`)

```python
async with aiohttp.ClientSession() as session:
    # Session is open
    response = await session.get(url)
    # Session automatically closes when exiting block
```

**Benefits:**
- Manages connection pooling
- Automatic resource cleanup
- Reuses TCP connections (faster!)

### 5. Sleeping (`asyncio.sleep`)

```python
# Bad - blocks entire program
import time
time.sleep(1)

# Good - lets other tasks run
await asyncio.sleep(1)
```

---

## Best Practices

### 1. Rate Limiting

```python
async def scrape_with_delay(urls):
    for url in urls:
        await fetch(url)
        await asyncio.sleep(0.5)  # Polite delay
```

### 2. Connection Pooling

```python
# Good - reuse session
async with aiohttp.ClientSession() as session:
    for url in urls:
        await fetch(session, url)

# Bad - create new session each time
for url in urls:
    async with aiohttp.ClientSession() as session:
        await fetch(session, url)
```

### 3. Error Handling

```python
async def safe_fetch(url):
    try:
        return await fetch(url)
    except aiohttp.ClientError as e:
        print(f"Network error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 4. Timeout Configuration

```python
timeout = aiohttp.ClientTimeout(total=30)
async with aiohttp.ClientSession(timeout=timeout) as session:
    await fetch(session, url)
```

### 5. Controlled Concurrency

```python
# Use semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

async def fetch_with_limit(session, url):
    async with semaphore:
        return await fetch(session, url)
```

---

## Common Pitfalls

### ❌ Forgetting `await`

```python
# Wrong - gets coroutine object, not result
result = async_function()

# Correct
result = await async_function()
```

### ❌ Using blocking I/O in async code

```python
# Bad - blocks event loop
def process_data():
    time.sleep(5)  # Blocks everything!
    
# Good - use async sleep
async def process_data():
    await asyncio.sleep(5)  # Lets other tasks run
```

### ❌ Creating new sessions repeatedly

```python
# Bad - overhead of creating new session each time
for url in urls:
    async with aiohttp.ClientSession() as session:
        await session.get(url)

# Good - reuse session
async with aiohttp.ClientSession() as session:
    for url in urls:
        await session.get(url)
```

### ❌ Not handling errors in concurrent tasks

```python
# If one task fails, gather() raises exception
results = await asyncio.gather(task1, task2, task3)

# Better - handle errors individually
async def safe_task(task):
    try:
        return await task
    except Exception as e:
        return {'error': str(e)}
```

---

## Performance Tuning

### Adjusting Batch Size

```python
# Conservative (polite, slower)
details = await scraper.scrape_multiple(links, batch_size=5)

# Aggressive (faster, might get blocked)
details = await scraper.scrape_multiple(links, batch_size=20)

# Recommended
details = await scraper.scrape_multiple(links, batch_size=10)
```

### Adding Request Delays

```python
# No delay - maximum speed (risky)
await asyncio.sleep(0)

# Small delay - balanced
await asyncio.sleep(0.5)

# Large delay - very polite (slower)
await asyncio.sleep(2)
```

### Connection Limits

```python
connector = aiohttp.TCPConnector(limit=10)  # Max 10 connections
async with aiohttp.ClientSession(connector=connector) as session:
    # ...
```

---

## Installation

```bash
# Install required package
pip install aiohttp

# Or add to requirements.txt
echo "aiohttp>=3.9.0" >> requirements.txt
pip install -r requirements.txt
```

---

## Summary

### Performance Gains

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| 10 pages scraping | 10s | 2s | 5x |
| 44 listings scraping | 88s | 10s | 8.8x |
| Total time | 98s | 12s | 8.2x |
| Memory usage | Low | Slightly higher | Acceptable |
| Code complexity | Simple | Moderate | Worth it |

### When to Use Async

✅ **Good for:**
- Web scraping (I/O-bound)
- API calls (network-bound)
- Database queries (I/O-bound)
- File I/O (reading/writing many files)

❌ **Not ideal for:**
- CPU-intensive tasks (use multiprocessing instead)
- Simple scripts with few requests
- When simplicity is more important than speed

### Key Takeaways

1. **Async = Non-blocking I/O** - work on other tasks while waiting
2. **Use `await`** - pauses function but not the program
3. **`asyncio.gather()`** - run multiple tasks concurrently
4. **Batching** - control concurrency to avoid overwhelming servers
5. **Connection pooling** - reuse `aiohttp.ClientSession`

---

## Further Reading

- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [Real Python: Async IO in Python](https://realpython.com/async-io-python/)
- [Understanding Async/Await](https://www.pythontutorial.net/python-concurrency/python-async-await/)

---

**Document created:** November 5, 2025  
**Author:** AI Assistant  
**Project:** Real Estate Market Analysis - Web Scraper
