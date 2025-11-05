import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import Any
import json
from pathlib import Path


class BaseScraper(ABC):
    """
    Abstract base class for web scraping.
    Provides common functionality for fetching and parsing HTML pages.
    """
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch HTML content from a URL asynchronously"""
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content into BeautifulSoup object"""
        return BeautifulSoup(html, 'html.parser')

    @abstractmethod
    async def scrape(self) -> Any:
        """Main scraping method - must be implemented by subclasses"""
        pass

    @abstractmethod
    def parse(self, soup: BeautifulSoup) -> Any:
        """Parse the BeautifulSoup object - must be implemented by subclasses"""
        pass


class LinkScraper(BaseScraper):
    """
    Scraper for extracting listing links from search results pages.
    Inherits from BaseScraper and implements pagination logic.
    """
    def __init__(self, num_pages: int = 5):
        super().__init__(base_url="https://www.sreality.cz")
        self.pages_url: str = f"{self.base_url}/hledani/prodej/byty/praha"
        self.num_pages: int = num_pages
        self.links: list[str] = []

    async def scrape(self) -> list[str]:
        """Scrape all listing links from multiple pages asynchronously"""
        self.links = []
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = []
            for page in range(1, self.num_pages + 1):
                url = f"{self.pages_url}?strana={page}"
                tasks.append(self._scrape_page(session, url))
            
            # Fetch all pages concurrently
            results = await asyncio.gather(*tasks)
            
            # Flatten the results
            for links in results:
                self.links.extend(links)
        
        self._clean_links()
        return self.links
    
    async def _scrape_page(self, session: aiohttp.ClientSession, url: str) -> list[str]:
        """Scrape a single page"""
        html = await self.fetch_page(session, url)
        soup = self.parse_html(html)
        return self.parse(soup)

    def parse(self, soup: BeautifulSoup) -> list[str]:
        """
        Parse a full search results page and return a list of links to individual listings
        """
        links: list = []

        # Primary selector for list items; fallback to id-based selector if empty
        li_nodes = soup.select('ul.MuiGrid2-root.MuiGrid2-container.MuiGrid2-direction-xs-row.css-1fesoy9 > li')
        if not li_nodes:
            li_nodes = soup.select('li[id^="region-tip-item"]')
        if not li_nodes:
            # Fallback: any <li> that contains the info container
            li_nodes = [li for li in soup.select('li') if li.select_one('div.css-adf8sc')]

        for li in li_nodes:
            parsed = self._parse_li_for_link(li)
            if parsed:
                links.append(parsed)

        return links

    def _parse_li_for_link(self, li_tag: BeautifulSoup) -> dict:
        """
        Parse single listing and return a link to the listing page itself    
        """
        # Find the <a> tag with the specific classes - use CSS selector with dots between classes
        link_element = li_tag.select_one('a.MuiTypography-root.MuiTypography-inherit.MuiLink-root.MuiLink-underlineAlways')
        
        # Alternative: you can also find any <a> tag with an href starting with /detail/
        if not link_element:
            link_element = li_tag.find('a', href=re.compile(r'^/detail/'))

        return link_element["href"] if link_element else None

    def _clean_links(self):
        """Convert relative URLs to absolute URLs"""
        for i in range(len(self.links)):
            if self.links[i] and not self.links[i].startswith(self.base_url):
                self.links[i] = self.base_url + self.links[i]


class DetailScraper(BaseScraper):
    """
    Scraper for extracting detailed information from individual listing pages.
    Inherits from BaseScraper and implements detail page parsing logic.
    """
    def __init__(self):
        super().__init__(base_url="https://www.sreality.cz")

    async def scrape(self, session: aiohttp.ClientSession, listing_url: str) -> dict[str, Any]:
        """Scrape detailed information from a single listing URL asynchronously"""
        html = await self.fetch_page(session, listing_url)
        soup = self.parse_html(html)
        return self.parse(soup, link=listing_url)

    def parse(self, soup: BeautifulSoup, link: str) -> dict[str, Any]:
        
        data = {}

        # --- Main info ---
        title = soup.select_one('h1.css-h2bhwn')
        title = title.get_text(strip=False) if title else None
        data['title'] = title
        
        # --- Details ---        
        description = soup.select_one('pre.css-16eb98b')
        data['popis'] = description.get_text(strip=True) if description else None

        keys = soup.select('dt.css-6g6jdp')
        values = soup.select('dd.css-1c70aha')

        details_dict = {}
        for key, value in zip(keys, values):
            key_text = key.get_text(strip=True).replace(':', '')
            value_text = value.get_text(strip=True)
            
            # Clean up zero-width spaces (U+200B) and other invisible characters
            value_text = value_text.replace('\u200b', '')
            
            details_dict[key_text] = value_text

        # remove součástí developerského projektu from details
        if 'Součástí developerského projektu' in details_dict:
            del details_dict['Součástí developerského projektu']

        for key, value in details_dict.items():
            data[key.lower().replace(' ', '_')] = value

        data['url'] = link

        data['link_id'] = link.split('/')[-1] if link else None

        return data

    async def scrape_multiple(self, listing_urls: list[str], batch_size: int = 10) -> list[dict[str, Any]]:
        """
        Scrape multiple listing URLs asynchronously with batching
        
        Args:
            listing_urls: List of URLs to scrape
            batch_size: Number of concurrent requests (default: 10)
        """
        results = []
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Process in batches to avoid overwhelming the server
            for i in range(0, len(listing_urls), batch_size):
                batch = listing_urls[i:i + batch_size]
                print(f"Scraping batch {i//batch_size + 1}/{(len(listing_urls)-1)//batch_size + 1} ({len(batch)} listings)")
                
                tasks = []
                for url in batch:
                    tasks.append(self._scrape_single_safe(session, url))
                
                # Wait for all tasks in this batch to complete
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                
                # Optional: Add a small delay between batches to be polite
                if i + batch_size < len(listing_urls):
                    await asyncio.sleep(0.5)
        
        return results
    
    async def _scrape_single_safe(self, session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
        """Scrape a single URL with error handling"""
        try:
            data = await self.scrape(session, url)
            return data
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {'url': url, 'error': str(e)}


async def main():
    """Main async function"""
    # Example usage: Scrape listing links
    print("Scraping listing links...")
    link_scraper = LinkScraper(num_pages=10)
    links = await link_scraper.scrape()
    print(f"Found {len(links)} links")
    
    # Example usage: Scrape details from listings
    if links:
        print("\nScraping details from listings...")
        detail_scraper = DetailScraper()
        details = await detail_scraper.scrape_multiple(links, batch_size=10)
        
        # Save to JSON
        path = Path(__file__).parent / 'listings.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(details, f, ensure_ascii=False, indent=4)
        
        print(f"\nSuccessfully scraped {len(details)} listings")
        print(f"Saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())