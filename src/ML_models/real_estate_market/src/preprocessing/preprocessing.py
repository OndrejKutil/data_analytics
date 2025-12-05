from pathlib import Path
import pandas as pd

SAVE_FILES = True  # Set to True to save preprocessed data to file

class Preprocessor:
    def __init__(self) -> None:
        self.base_path = Path(__file__).parent.parent
        self.input_path = self.base_path / 'scraper' / 'listings.csv'
        self.output_path = self.base_path / 'preprocessing' / 'preprocessed_listings.csv'
    
    def _safe_get_str(self, data: dict[str, str | None | float | bool | int], key: str, default: str | None | float = None) -> str | None | float:
        """Safely get string value, handling NaN from CSV"""
        value = data.get(key, default)
        if isinstance(value, float) and pd.isna(value):
            return default
        return value

    def _extract_features_from_description(self, description: str) -> dict[str, bool]:
        if not description or pd.isna(description):
            return {}
        
        desc_lower = str(description).lower()
        
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

    def _extract_features(self, data: dict[str, str | None | float | bool | int]) -> dict[str, str | None | float | bool | int]:
        """
        Extracts all features into a flat structure for CSV output.
        """
        features = {}
        
        # Basic info
        features['price_czk'] = data.get('price_czk')
        
        # Location
        features['city'] = self._safe_get_str(data, 'city')
        features['district'] = self._safe_get_str(data, 'district')
        features['street'] = self._safe_get_str(data, 'street')
        features['zip_code'] = self._safe_get_str(data, 'zip')
        
        # Property details
        features['layout'] = self._safe_get_str(data, 'category_sub') # e.g. 3+kk
        features['building_type'] = self._safe_get_str(data, 'building_type') # Panelová, Cihlová
        features['condition'] = self._safe_get_str(data, 'building_condition')
        features['ownership'] = self._safe_get_str(data, 'ownership')
        features['floor'] = data.get('floor_number')
        features['total_floors'] = data.get('floors')
        
        # Areas
        features['usable_area'] = data.get('usable_area') or data.get('floor_area')
        features['balcony_area'] = data.get('balcony_area')
        features['terrace_area'] = data.get('terrace_area')
        features['loggia_area'] = data.get('loggia_area')
        features['cellar_area'] = data.get('cellar_area')
        
        # Boolean/Categorical features from structured data
        balcony_area = data.get('balcony_area')
        features['has_balcony'] = bool(data.get('balcony')) or (isinstance(balcony_area, (int, float)) and balcony_area > 0)
        terrace_area = data.get('terrace_area')
        features['has_terrace'] = bool(data.get('terrace')) or (isinstance(terrace_area, (int, float)) and terrace_area > 0)
        loggia_area = data.get('loggia_area')
        features['has_loggia'] = bool(data.get('loggia')) or (isinstance(loggia_area, (int, float)) and loggia_area > 0)
        cellar_area = data.get('cellar_area')
        features['has_cellar'] = bool(data.get('cellar')) or (isinstance(cellar_area, (int, float)) and cellar_area > 0)
        garage_count = data.get('garage_count')
        features['has_garage'] = bool(data.get('garage')) or (isinstance(garage_count, (int, float)) and garage_count > 0)
        features['has_parking'] = bool(data.get('parking_lots'))
        elevator = self._safe_get_str(data, 'elevator')
        features['has_elevator'] = True if elevator == 'Ano' else False
        
        # Energy
        energy_rating = data.get('energy_efficiency_rating')
        if energy_rating and isinstance(energy_rating, str):
             features['energy_rating'] = energy_rating.split(' ')[0]
        else:
             features['energy_rating'] = None

        return features



    def _preprocess_one_json(self, raw_data: dict[str, str | None | float | bool | int]) -> dict[str, str | None | float | bool | int]:
        # 1. Extract features
        features = self._extract_features(raw_data)
        
        # 2. Extract from description
        description = raw_data.get('description', '')
        # Handle NaN from CSV
        if isinstance(description, float) and pd.isna(description):
            description = ''
        # Ensure description is a string
        if not isinstance(description, str):
            description = str(description) if description is not None else ''
        desc_features = self._extract_features_from_description(description)
        features.update(desc_features)
        
        # 3. Consolidate renovation status
        if features['condition'] == 'Po rekonstrukci':
            features['is_renovated'] = True
        elif features.get('desc_is_renovated'):
            features['is_renovated'] = True
        else:
            features['is_renovated'] = False
            
        # 4. Consolidate new building status
        if features['condition'] == 'Novostavba' or features.get('desc_is_new_building'):
             features['is_new_building'] = True
        else:
             features['is_new_building'] = False

        return features

    def run(self) -> list[dict[str, str | bool | float | int | None]]:
        if not self.input_path.exists():
            print(f"Input file not found: {self.input_path}")
            return []

        print(f"Loading data from {self.input_path}...")
        raw_df = pd.read_csv(self.input_path, encoding='utf-8')
        
        print(f"Preprocessing {len(raw_df)} listings...")
        preprocessed_data = []
        for idx, row in raw_df.iterrows():
            try:
                # Convert row to dict
                item = row.to_dict()
                processed = self._preprocess_one_json(item)
                if processed:
                    preprocessed_data.append(processed)
            except Exception as e:
                print(f"Error processing item at index {idx}: {e}")
                continue

        print(f"Successfully preprocessed {len(preprocessed_data)} listings.")

        if SAVE_FILES:
            print(f"Saving to {self.output_path}...")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(preprocessed_data)
            df.to_csv(self.output_path, index=False, encoding='utf-8')
            print(f"Saved {len(df)} listings to CSV")
        
        return preprocessed_data

if __name__ == "__main__":
    preprocessor: Preprocessor = Preprocessor()
    preprocessor.run()

