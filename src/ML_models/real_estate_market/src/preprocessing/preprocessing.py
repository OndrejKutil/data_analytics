import json
from pathlib import Path
from typing import Dict

SAVE_FILES = True  # Set to True to save preprocessed data to file

class Preprocessor:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.input_path = self.base_path / 'scraper' / 'listings.json'
        self.output_path = self.base_path / 'preprocessing' / 'preprocessed_listings.json'

    def _extract_features_from_description(self, description: str) -> Dict[str, bool]:
        if not description:
            return {}
        
        desc_lower = description.lower()
        
        features = {
            'desc_has_metro': any(x in desc_lower for x in ['metro', 'stanice metra']),
            'desc_has_tram': 'tram' in desc_lower,
            'desc_has_bus': 'autobus' in desc_lower or 'bus' in desc_lower,
            'desc_has_park': 'park' in desc_lower or 'les' in desc_lower,
            'desc_has_school': 'škol' in desc_lower, # škola, školka
            'desc_has_shopping': any(x in desc_lower for x in ['obchod', 'nákup', 'supermarket', 'potraviny']),
            'desc_is_quiet': any(x in desc_lower for x in ['tichý', 'klid', 'klidná']),
            'desc_is_sunny': any(x in desc_lower for x in ['světlý', 'slunný', 'prosluněný']),
            'desc_has_ac': 'klimatiza' in desc_lower,
            'desc_has_fireplace': 'krb' in desc_lower,
            'desc_has_floor_heating': 'podlahové topení' in desc_lower or 'podlahové vytápění' in desc_lower,
            'desc_is_renovated': 'rekonstruk' in desc_lower,
            'desc_is_new_building': 'novostavb' in desc_lower,
        }
        return features

    def _extract_model_data(self, data: Dict) -> Dict:
        """
        Extracts features relevant for model training (price, location, physical attributes).
        """
        model_data = {}
        
        # Basic info
        model_data['price_czk'] = data.get('price_czk')
        
        # Location
        model_data['city'] = data.get('city')
        model_data['district'] = data.get('district')
        model_data['street'] = data.get('street')
        model_data['zip_code'] = data.get('zip')
        
        # Property details
        model_data['layout'] = data.get('category_sub') # e.g. 3+kk
        model_data['building_type'] = data.get('building_type') # Panelová, Cihlová
        model_data['condition'] = data.get('building_condition')
        model_data['ownership'] = data.get('ownership')
        model_data['floor'] = data.get('floor_number')
        model_data['total_floors'] = data.get('floors')
        
        # Areas
        model_data['usable_area'] = data.get('usable_area') or data.get('floor_area')
        model_data['balcony_area'] = data.get('balcony_area')
        model_data['terrace_area'] = data.get('terrace_area')
        model_data['loggia_area'] = data.get('loggia_area')
        model_data['cellar_area'] = data.get('cellar_area')
        
        # Boolean/Categorical features from structured data
        model_data['has_balcony'] = bool(data.get('balcony')) or (data.get('balcony_area') is not None and data.get('balcony_area', 0) > 0)
        model_data['has_terrace'] = bool(data.get('terrace')) or (data.get('terrace_area') is not None and data.get('terrace_area', 0) > 0)
        model_data['has_loggia'] = bool(data.get('loggia')) or (data.get('loggia_area') is not None and data.get('loggia_area', 0) > 0)
        model_data['has_cellar'] = bool(data.get('cellar')) or (data.get('cellar_area') is not None and data.get('cellar_area', 0) > 0)
        model_data['has_garage'] = bool(data.get('garage')) or (data.get('garage_count') is not None and data.get('garage_count', 0) > 0)
        model_data['has_parking'] = bool(data.get('parking_lots'))
        model_data['has_elevator'] = True if data.get('elevator') == 'Ano' else False
        
        # Energy
        if data.get('energy_efficiency_rating'):
             model_data['energy_rating'] = data.get('energy_efficiency_rating', '').split(' ')[0]
        else:
             model_data['energy_rating'] = None

        return model_data

    def _extract_metadata(self, data: Dict) -> Dict:
        """
        Extracts metadata and other useful info not directly used for model training.
        """
        metadata = {}
        metadata['title'] = data.get('name')
        metadata['price_czk_per_sqm'] = data.get('price_czk_per_sqm')
        metadata['lat'] = data.get('latitude')
        metadata['lon'] = data.get('longitude')
        metadata['seller_id'] = data.get('seller_id')
        metadata['seller_name'] = data.get('seller_name')
        metadata['seller_email'] = data.get('seller_email')
        metadata['seller_phones'] = data.get('seller_phones')
        metadata['premise_name'] = data.get('premise_name')
        metadata['premise_web_url'] = data.get('premise_web_url')
        metadata['images_count'] = data.get('images_count')
        metadata['image_urls'] = [img.get('url') for img in data.get('images', [])][:5]
        metadata['listing_url'] = data.get('url')
        metadata['listing_id'] = data.get('listing_id')
        metadata['scraped_at'] = data.get('scraped_at')

        # Note: Scrape date is not currently in the raw data, but would go here.
        
        return metadata

    def _preprocess_one_json(self, raw_data: dict) -> dict:
        # 1. Extract Model Data
        model_data = self._extract_model_data(raw_data)
        
        # 2. Extract from description (add to model data)
        description = raw_data.get('description', '')
        desc_features = self._extract_features_from_description(description)
        model_data.update(desc_features)
        
                
        # Consolidate renovation status
        if model_data['condition'] == 'Po rekonstrukci':
            model_data['is_renovated'] = True
        elif model_data.get('desc_is_renovated'):
            model_data['is_renovated'] = True
        else:
            model_data['is_renovated'] = False
            
        # Consolidate new building status
        if model_data['condition'] == 'Novostavba' or model_data.get('desc_is_new_building'):
             model_data['is_new_building'] = True
        else:
             model_data['is_new_building'] = False

        # 4. Extract Metadata
        metadata = self._extract_metadata(raw_data)

        return {
            'model_training_data': model_data,
            'metadata': metadata
        }

    def run(self):
        if not self.input_path.exists():
            print(f"Input file not found: {self.input_path}")
            return

        print(f"Loading data from {self.input_path}...")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        print(f"Preprocessing {len(raw_data)} listings...")
        preprocessed_data = []
        for item in raw_data:
            try:
                processed = self._preprocess_one_json(item)
                if processed:
                    preprocessed_data.append(processed)
            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        print(f"Successfully preprocessed {len(preprocessed_data)} listings.")

        if SAVE_FILES:
            print(f"Saving to {self.output_path}...")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)
        
        return preprocessed_data

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()

