import re
import requests
from bs4 import BeautifulSoup
import pprint
import pandas as pd

class Scraper:
    def __init__(self, num_pages: int = 5):
        self.base_url = "https://www.sreality.cz/hledani/prodej/byty/praha"
        self.listings = []
        self.num_pages = num_pages

    def return_listings(self):
        self._scrape_listings(max_pages=self.num_pages)
        return self.listings
    
    def save_listings_csv(self):
        self._scrape_listings(max_pages=self.num_pages)
        df = pd.DataFrame(self.listings)
        df.to_csv('listings.csv', index=False)


    def _scrape_listings(self, max_pages: int = 5) -> list:
        for page in range(1, max_pages + 1):
            html = self._fetch_page(page)
            listings = self._parse_page(html)

            self.listings.extend(listings)

    def _fetch_page(self, page_number: int) -> str:
        url = f"{self.base_url}?strana={page_number}"
        response = requests.get(url)
        response.raise_for_status()

        return response.text

    def _parse_page(self, html: str) -> list:
        """
        Parse a full search results page and return a list of listings with
        location, price_czk (int or None), and area_sqm (float or None).
        """
        listings = []
        soup = BeautifulSoup(html, 'html.parser')

        # Primary selector for list items; fallback to id-based selector if empty
        li_nodes = soup.select('ul.MuiGrid2-root.MuiGrid2-container.MuiGrid2-direction-xs-row.css-1fesoy9 > li')
        if not li_nodes:
            li_nodes = soup.select('li[id^="region-tip-item"]')
        if not li_nodes:
            # Fallback: any <li> that contains the info container
            li_nodes = [li for li in soup.select('li') if li.select_one('div.css-adf8sc')]

        for li in li_nodes:
            parsed = self._parse_listing_li(li)
            if parsed:
                listings.append(parsed)

        return listings

    @staticmethod
    def _parse_listing_li(li_tag: BeautifulSoup) -> dict:
        """
        Extract data from a single listing <li> BeautifulSoup tag.
        Returns dict with keys: location, price_text, price_czk, area_sqm.
        """
        info = li_tag.select_one('div.css-adf8sc')
        if not info:
            # Sometimes the container might be under an anchor
            info = li_tag.select_one('a div.css-adf8sc')
        if not info:
            return None

        # There are typically two paragraphs with class css-d7upve
        text_blocks = [p.get_text(strip=True) for p in info.select('p.css-d7upve')]
        title_text = text_blocks[0] if text_blocks else ''
        location = text_blocks[1] if len(text_blocks) > 1 else ''

        # Price element may be <p> or <div> with class css-ca9wwd
        price_el = info.select_one('.css-ca9wwd')
        price_text = price_el.get_text(strip=True) if price_el else ''

        # Parse price numeric (CZK); keep None if not available (e.g., Cena dohodou)
        price_digits = re.sub(r'\D', '', price_text)
        price_czk = int(price_digits) if price_digits else None

        # Parse area (square meters) from title like "Prodej bytu 4+kk 110 m²"
        area_sqm = None
        m2_match = re.search(r'(\d+[\s\u00A0\d]*[\.,]?\d*)\s*m(?:\u00B2|2)', title_text)
        if m2_match:
            # Normalize: remove spaces, replace comma with dot
            raw = m2_match.group(1).replace('\u00A0', '').replace(' ', '').replace('\xa0', '')
            raw = raw.replace(',', '.')
            try:
                area_sqm = float(raw)
            except ValueError:
                area_sqm = None

        return {
            'steet': location.split(',')[0] if ',' in location else location,
            'prague_location': location.split(',')[1] if ',' in location else location,
            'price_czk': price_czk,
            'area_sqm': area_sqm,
        }

scraper = Scraper(num_pages=100)
scraper.save_listings_csv()