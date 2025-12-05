import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup, Tag
from abc import ABC, abstractmethod
from typing import Any
import json
from pathlib import Path
import pandas as pd

PAGES_TO_SCRAPE = None  # Set to None to scrape all available pages
SAVE_FILES = True  # Set to True to save scraped data to file
BATCH_SIZE = 30  # Number of concurrent requests when scraping details
# Set save path for incremental saves if SAVE_FILES is enabled
SAVE_PATH = Path(__file__).parent / 'listings.csv' if SAVE_FILES else None


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
    async def scrape(self, *args: Any, **kwargs: Any) -> Any:
        """Main scraping method - must be implemented by subclasses"""
        pass

    @abstractmethod
    def parse(self, soup: BeautifulSoup, *args: Any, **kwargs: Any) -> Any:
        """Parse the BeautifulSoup object - must be implemented by subclasses"""
        pass


class LinkScraper(BaseScraper):
    """
    Scraper for extracting listing links from search results pages.
    Inherits from BaseScraper and implements pagination logic.
    """
    def __init__(self, num_pages: int | None = 5):
        super().__init__(base_url="https://www.sreality.cz")
        self.pages_url: str = f"{self.base_url}/hledani/prodej/byty/praha"
        self.num_pages: int | None = num_pages
        self.links: list[str] = []

    async def scrape(self) -> list[str]:
        """Scrape all listing links from multiple pages asynchronously"""
        self.links = []
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # If num_pages is None, detect total pages automatically
            if self.num_pages is None:
                self.num_pages = await self._detect_total_pages(session)
                print(f"Detected {self.num_pages} total pages to scrape")
            
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
        try:
            html = await self.fetch_page(session, url)
            soup = self.parse_html(html)
            return self.parse(soup)
        except Exception as e:
            print(f"\nWarning: Failed to scrape {url}: {e}")
            return []
    
    async def _detect_total_pages(self, session: aiohttp.ClientSession) -> int:
        """Detect the total number of pages available by binary search"""
        try:
            # First, get an upper bound estimate from the first page
            url = f"{self.pages_url}?strana=1"
            html = await self.fetch_page(session, url)
            soup = self.parse_html(html)
            
            # Try to find estimated max from pagination links
            pagination = soup.select('a[href*="strana="]')
            estimated_max = 1
            
            for link in pagination:
                href = link.get('href', '')
                if isinstance(href, str):
                    match = re.search(r'strana=(\d+)', href)
                    if match:
                        estimated_max = max(estimated_max, int(match.group(1)))
            
            # Fallback: try to find text like "strana 1 z 50"
            if estimated_max == 1:
                page_text = soup.get_text()
                match = re.search(r'strana \d+ z (\d+)', page_text, re.IGNORECASE)
                if match:
                    estimated_max = int(match.group(1))
            
            if estimated_max == 1:
                print("Warning: Could not detect pagination, defaulting to 1 page")
                return 1
            
            # Now verify the actual last page exists using binary search
            # This handles cases where pagination shows pages that don't exist
            left, right = 1, estimated_max
            last_valid_page = 1
            
            while left <= right:
                mid = (left + right) // 2
                test_url = f"{self.pages_url}?strana={mid}"
                
                try:
                    async with session.get(test_url) as response:
                        if response.status == 200:
                            # Check if page has listings (not empty)
                            html = await response.text()
                            soup = self.parse_html(html)
                            links = self.parse(soup)
                            
                            if links:  # Page exists and has listings
                                last_valid_page = mid
                                left = mid + 1
                            else:  # Page exists but empty
                                right = mid - 1
                        else:  # 404 or other error
                            right = mid - 1
                except:
                    right = mid - 1
            
            return last_valid_page
            
        except Exception as e:
            print(f"Error detecting total pages: {e}. Defaulting to 1 page")
            return 1

    def parse(self, soup: BeautifulSoup) -> list[str]:
        """
        Parse a full search results page and return a list of links to individual listings
        """
        links: list[str] = []

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

    def _parse_li_for_link(self, li_tag: Tag) -> str | None:
        """
        Parse single listing and return a link to the listing page itself    
        """
        # Find the <a> tag with the specific classes - use CSS selector with dots between classes
        link_element = li_tag.select_one('a.MuiTypography-root.MuiTypography-inherit.MuiLink-root.MuiLink-underlineAlways')
        
        # Alternative: you can also find any <a> tag with an href starting with /detail/
        if not link_element:
            link_element = li_tag.find('a', href=re.compile(r'^/detail/'))

        if link_element and isinstance(link_element, Tag):
            href = link_element.get("href")
            return str(href) if href else None
        return None

    def _clean_links(self) -> None:
        """Convert relative URLs to absolute URLs"""
        for i in range(len(self.links)):
            if self.links[i] and not self.links[i].startswith(self.base_url):
                self.links[i] = self.base_url + self.links[i]


class DetailScraper(BaseScraper):
    """
    Scraper for extracting detailed information from individual listing pages.
    Inherits from BaseScraper and implements detail page parsing logic.
    """
    def __init__(self) -> None:
        super().__init__(base_url="https://www.sreality.cz")

    async def scrape(self, session: aiohttp.ClientSession, listing_url: str) -> dict[str, Any]:
        """Scrape detailed information from a single listing URL asynchronously"""
        html = await self.fetch_page(session, listing_url)
        # Uncomment below to debug HTML structure
        # print()
        # print(html)
        # print()
        soup = self.parse_html(html)
        return self.parse(soup, link=listing_url)

    def parse(self, soup: BeautifulSoup, link: str) -> dict[str, Any]:
        """
        Parse the listing page and extract all available data.
        Priority: Extract from JSON embedded in page, fallback to HTML scraping.
        """
        from datetime import datetime
        
        data = {}
        
        # Try to extract JSON data from __NEXT_DATA__ script tag (most reliable)
        json_data = None
        script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        if script_tag and isinstance(script_tag, Tag) and script_tag.string:
            try:
                json_data = json.loads(script_tag.string)
                estate_data = json_data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
                
                if not estate_data:
                    raise ValueError("No estate_data found in JSON")
                
                # Find the estate query
                for query in estate_data:
                    if 'estate' in query.get('queryKey', []):
                        estate_info = query.get('state', {}).get('data', {})
                        if not estate_info:
                            continue
                            
                        try:
                            # Extract all available fields from the structured data
                            
                            # Basic info
                            data['name'] = estate_info.get('name')
                            data['description'] = estate_info.get('description')
                            data['note'] = estate_info.get('note')
                            
                            # Price information
                            data['price'] = estate_info.get('price')
                            data['price_czk'] = estate_info.get('priceCzk')
                            data['price_czk_per_sqm'] = estate_info.get('priceCzkPerSqM')
                            data['price_summary_czk'] = estate_info.get('priceSummaryCzk')
                            data['price_summary_old_czk'] = estate_info.get('priceSummaryOldCzk')
                            
                            # Price currency and unit
                            price_currency = estate_info.get('priceCurrencyCb') or {}
                            data['price_currency'] = price_currency.get('name')
                            
                            price_unit = estate_info.get('priceUnitCb') or {}
                            data['price_unit'] = price_unit.get('name')
                            
                            # Category information
                            category_main = estate_info.get('categoryMainCb') or {}
                            data['category_main'] = category_main.get('name')
                            
                            category_sub = estate_info.get('categorySubCb') or {}
                            data['category_sub'] = category_sub.get('name')
                            
                            category_type = estate_info.get('categoryTypeCb') or {}
                            data['category_type'] = category_type.get('name')
                            
                            # Location data
                            locality = estate_info.get('locality') or {}
                            data['latitude'] = locality.get('latitude')
                            data['longitude'] = locality.get('longitude')
                            data['city'] = locality.get('city')
                            data['city_part'] = locality.get('cityPart')
                            data['district'] = locality.get('district')
                            data['region'] = locality.get('region')
                            data['street'] = locality.get('street')
                            data['street_number'] = locality.get('streetNumber')
                            data['house_number'] = locality.get('houseNumber')
                            data['zip'] = locality.get('zip')
                            data['ward'] = locality.get('ward')
                            data['quarter'] = locality.get('quarter')
                            data['municipality'] = locality.get('municipality')
                            data['inaccuracy_type'] = locality.get('inaccuracyType')
                        except Exception as e:
                            print(f"Warning: Error extracting basic/location data: {e}")
                        
                        try:
                            # Images
                            images = estate_info.get('images', [])
                            image_urls = []
                            for img in images:
                                img_url = img.get('url', '')
                                if img_url:
                                    # Make URL absolute
                                    if img_url.startswith('//'):
                                        img_url = 'https:' + img_url
                                    image_urls.append({
                                        'url': img_url,
                                        'order': img.get('order'),
                                        'width': img.get('width'),
                                        'height': img.get('height')
                                    })
                            data['images'] = image_urls
                            data['images_count'] = len(image_urls)
                            
                            # Videos
                            videos = estate_info.get('videos', [])
                            data['videos'] = videos
                            data['has_videos'] = len(videos) > 0
                            
                            # Panorama
                            data['has_panorama'] = estate_info.get('panorama', False)
                            data['matterport_url'] = estate_info.get('matterportUrl')
                        except Exception as e:
                            print(f"Warning: Error extracting media data: {e}")
                        
                        try:
                            # Agency/Seller information
                            seller = estate_info.get('seller') or {}
                            data['seller_id'] = seller.get('id')
                            data['seller_name'] = seller.get('name')
                            data['seller_email'] = seller.get('email')
                            data['seller_image'] = seller.get('image')
                            phones = seller.get('phones', [])
                            data['seller_phones'] = [p.get('phone') for p in phones] if phones else []
                            
                            premise = estate_info.get('premise') or {}
                            data['premise_id'] = premise.get('id')
                            data['premise_name'] = premise.get('name')
                            data['premise_ico'] = premise.get('ico')
                            data['premise_logo'] = premise.get('logo')
                            data['premise_web_url'] = premise.get('webUrl')
                            data['premise_review_count'] = premise.get('reviewCount')
                            data['premise_review_score'] = premise.get('reviewScore')
                        except Exception as e:
                            print(f"Warning: Error extracting seller data: {e}")
                        
                        try:
                            # Property parameters
                            params = estate_info.get('params') or {}
                            
                            # Building details
                            data['building_type'] = (params.get('buildingType') or {}).get('name')
                            data['building_condition'] = (params.get('buildingCondition') or {}).get('name')
                            data['building_area'] = params.get('buildingArea')
                            data['acceptance_year'] = params.get('acceptanceYear')
                            data['reconstruction_year'] = params.get('reconstructionYear')
                            
                            # Floor information
                            data['floor_number'] = params.get('floorNumber')
                            data['floors'] = params.get('floors')
                            data['underground_floors'] = params.get('undergroundFloors')
                            
                            # Areas
                            data['floor_area'] = params.get('floorArea')
                            data['usable_area'] = params.get('usableArea')
                            data['balcony'] = params.get('balcony', False)
                            data['balcony_area'] = params.get('balconyArea')
                            data['terrace'] = params.get('terrace', False)
                            data['terrace_area'] = params.get('terraceArea')
                            data['loggia'] = params.get('loggia', False)
                            data['loggia_area'] = params.get('loggiaArea')
                            data['cellar'] = params.get('cellar', False)
                            data['cellar_area'] = params.get('cellarArea')
                            data['garden_area'] = params.get('gardenArea')
                            data['basin'] = params.get('basin', False)
                            data['basin_area'] = params.get('basinArea')
                            
                            # Amenities
                            data['elevator'] = (params.get('elevator') or {}).get('name')
                            data['garage'] = params.get('garage', False)
                            data['garage_count'] = params.get('garageCount')
                            data['parking_lots'] = params.get('parkingLots', False)
                            data['furnished'] = (params.get('furnished') or {}).get('name')
                            data['garret'] = params.get('garret', False)
                            
                            # Ownership and legal
                            data['ownership'] = (params.get('ownership') or {}).get('name')
                            
                            # Energy and utilities
                            data['energy_efficiency_rating'] = (params.get('energyEfficiencyRating') or {}).get('name')
                            data['energy_performance_certificate'] = (params.get('energyPerformanceCertificate') or {}).get('name')
                            data['low_energy'] = params.get('lowEnergy', False)
                        except Exception as e:
                            print(f"Warning: Error extracting property parameters: {e}")
                        
                        try:
                            # Heating
                            params = estate_info.get('params') or {}
                            heating_set = params.get('heatingSet', [])
                            data['heating_types'] = [h.get('name') for h in heating_set if h] if heating_set else []
                            
                            heating_source_set = params.get('heatingSourceSet', [])
                            data['heating_sources'] = [h.get('name') for h in heating_source_set if h] if heating_source_set else []
                            
                            heating_element_set = params.get('heatingElementSet', [])
                            data['heating_elements'] = [h.get('name') for h in heating_element_set if h] if heating_element_set else []
                            
                            water_heat_source_set = params.get('waterHeatSourceSet', [])
                            data['water_heat_sources'] = [w.get('name') for w in water_heat_source_set if w] if water_heat_source_set else []
                            
                            # Utilities
                            water_set = params.get('waterSet', [])
                            data['water_types'] = [w.get('name') for w in water_set if w] if water_set else []
                            
                            gully_set = params.get('gullySet', [])
                            data['sewage_types'] = [g.get('name') for g in gully_set if g] if gully_set else []
                            
                            gas_set = params.get('gasSet', [])
                            data['gas_types'] = [g.get('name') for g in gas_set if g] if gas_set else []
                            
                            electricity_set = params.get('electricitySet', [])
                            data['electricity_types'] = [e.get('name') for e in electricity_set if e] if electricity_set else []
                            
                            # Telecommunications
                            telecom_set = params.get('telecommunicationSet', [])
                            data['telecommunication_types'] = [t.get('name') for t in telecom_set if t] if telecom_set else []
                            
                            internet_connection_set = params.get('internetConnectionTypeSet', [])
                            data['internet_connection_types'] = [i.get('name') for i in internet_connection_set if i] if internet_connection_set else []
                            data['internet_provider'] = params.get('internetConnectionProvider')
                            data['internet_speed'] = params.get('internetConnectionSpeed')
                            
                            # Transport
                            transport_set = params.get('transportSet', [])
                            data['transport_types'] = [t.get('name') for t in transport_set if t] if transport_set else []
                            
                            road_type_set = params.get('roadTypeSet', [])
                            data['road_types'] = [r.get('name') for r in road_type_set if r] if road_type_set else []
                            
                            # Location characteristics
                            data['object_location'] = (params.get('objectLocation') or {}).get('name')
                            data['surroundings_type'] = (params.get('surroundingsType') or {}).get('name')
                            
                            # Dates
                            data['ready_date'] = params.get('readyDate')
                            data['since'] = params.get('since')
                            data['edited'] = params.get('edited')
                            data['beginning_date'] = params.get('beginningDate')
                            data['finish_date'] = params.get('finishDate')
                        except Exception as e:
                            print(f"Warning: Error extracting utilities/transport data: {e}")
                        
                        try:
                            # Additional flags
                            data['is_exclusively'] = estate_info.get('isExclusively', False)
                            
                            # Nearby POIs (Points of Interest)
                            nearest = estate_info.get('nearest', [])
                            data['nearby_pois_count'] = len(nearest)
                            
                            # Extended POIs by category
                            extended_pois = estate_info.get('extendedPois') or {}
                            if extended_pois:
                                data['nearby_transport'] = len((extended_pois.get('transport') or {}).get('values', []))
                                data['nearby_doctors'] = len((extended_pois.get('doctors') or {}).get('values', []))
                                data['nearby_grocery'] = len((extended_pois.get('grocery') or {}).get('values', []))
                                data['nearby_leisure'] = len((extended_pois.get('leisure') or {}).get('values', []))
                                data['nearby_restaurants'] = len((extended_pois.get('restaurants') or {}).get('values', []))
                                data['nearby_schools'] = len((extended_pois.get('schools') or {}).get('values', []))
                        except Exception as e:
                            print(f"Warning: Error extracting POI data: {e}")
                            
                        break
                        
            except Exception as e:
                print(f"Warning: Could not parse JSON structure: {e}")
                # Will fall back to HTML parsing
        
        # Fallback: If JSON parsing failed, extract from HTML
        if not data or len(data) < 5:
            # Basic info from HTML
            title = soup.select_one('h1.css-h2bhwn, h1')
            data['name'] = title.get_text(strip=True) if title else None
            
            description = soup.select_one('pre.css-16eb98b')
            data['description'] = description.get_text(strip=True) if description else None
            
            # Price from HTML
            price_element = soup.select_one('span.css-105e6fq, strong.css-105e6fq')
            if price_element:
                price_text = price_element.get_text(strip=True)
                price_clean = ''.join(filter(str.isdigit, price_text))
                data['price'] = int(price_clean) if price_clean else None
            
            # Extract details from dt/dd pairs
            keys = soup.select('dt.css-6g6jdp')
            values = soup.select('dd.css-1c70aha')
            
            for key, value in zip(keys, values):
                key_text = key.get_text(strip=True).replace(':', '').lower().replace(' ', '_')
                value_text = value.get_text(strip=True).replace('\u200b', '')
                data[f'html_{key_text}'] = value_text
        
        # Add scraping metadata
        data['url'] = link
        data['listing_id'] = link.split('/')[-1] if link else None
        data['scraped_at'] = datetime.now().isoformat()
        
        return data

    async def scrape_multiple(self, listing_urls: list[str], batch_size: int = 10, 
                             save_path: Path | None = None) -> list[dict[str, Any]]:
        """
        Scrape multiple listing URLs asynchronously with batching and incremental saves
        
        Args:
            listing_urls: List of URLs to scrape
            batch_size: Number of concurrent requests (default: 10)
            save_path: Path to save incremental results (optional). If provided, saves after each batch.
        """
        results = []
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Process in batches to avoid overwhelming the server
            total_batches = (len(listing_urls) - 1) // batch_size + 1
            for i in range(0, len(listing_urls), batch_size):
                batch = listing_urls[i:i + batch_size]
                batch_num = i // batch_size + 1
                print(f"""\rScraping batch {batch_num}/{total_batches} (Total listings {len(listing_urls)})""", end='', flush=True)
                
                tasks = []
                for url in batch:
                    tasks.append(self._scrape_single_safe(session, url))
                
                # Wait for all tasks in this batch to complete
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                
                # Incremental save after each batch
                if save_path and SAVE_FILES:
                    self._save_results(results, save_path)
                
                # Optional: Add a small delay between batches to be polite
                if i + batch_size < len(listing_urls):
                    await asyncio.sleep(0.5)
            
            # Clear the progress line
            print('\r' + ' ' * 80 + '\r', end='', flush=True)
        
        return results
    
    async def _scrape_single_safe(self, session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
        """Scrape a single URL with error handling"""
        try:
            data = await self.scrape(session, url)
            return data
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {'url': url, 'error': str(e)}
    
    def _save_results(self, results: list[dict[str, Any]], save_path: Path) -> None:
        """Save results to CSV file with proper formatting"""
        if not results:
            return
            
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(results)
        
        # Convert list columns to JSON strings for CSV compatibility
        list_columns = ['images', 'videos', 'seller_phones', 'heating_types', 'heating_sources', 
                       'heating_elements', 'water_heat_sources', 'water_types', 'sewage_types', 
                       'gas_types', 'electricity_types', 'telecommunication_types', 
                       'internet_connection_types', 'transport_types', 'road_types']
        
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
        
        # Save to CSV (overwrites previous version with updated data)
        df.to_csv(save_path, index=False, encoding='utf-8')


class Scraper():
    def __init__(self) -> None:
        pass

    async def scrape_sreality(self, default_pages_amount: int | None = PAGES_TO_SCRAPE) -> list[dict[str, str]]:
        """Scrape real estate listings from sreality.cz

        Args:
            default_pages_amount (int | None, optional): Number of pages to scrape. 
                Defaults to PAGES_TO_SCRAPE. If None, scrapes all available pages.
                If passed value <=0, uses PAGES_TO_SCRAPE.

        Returns:
            list[dict[str, str]]: List of scraped listing details
        """

        if default_pages_amount is not None and default_pages_amount <= 0:
            default_pages_amount = PAGES_TO_SCRAPE

        link_scraper = LinkScraper(num_pages=default_pages_amount)
        links = await link_scraper.scrape()
        
        # Example usage: Scrape details from listings
        if links:

            detail_scraper = DetailScraper()

            
            details = await detail_scraper.scrape_multiple(links, batch_size=BATCH_SIZE, save_path=SAVE_PATH)
            
            if SAVE_FILES and SAVE_PATH:
                print(f"Saved to: {SAVE_PATH}")
            
            print(f"Successfully scraped {len(details)} listings")

            return details

        else:
            return []
        


if __name__ == "__main__":
    scraper = Scraper()
    data = asyncio.run(scraper.scrape_sreality(default_pages_amount=PAGES_TO_SCRAPE))