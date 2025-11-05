import json
from pathlib import Path
import re

# Get the path relative to the preprocessing.py file
json_path = Path(__file__).parent.parent / 'scraper' / 'listings.json'

with open(json_path, 'r', encoding='utf-8') as f:
    raw_json = json.load(f)

def preprocess_data(raw_data: dict) -> dict:

    data = raw_data.copy()

    # Skip if title is None or missing
    if not data.get('title'):
        return None
    
    title = data['title'].split('m²')

    location = title[1]
    title = title[0].split(' ')
    rooms = title[-2]
    meters = title[-1].replace('\xa0', '')

    data['lokalita'] = location
    data['část_prahy'] = location.split('-')[1].strip() if '-' in location else None
    data['adresa'] = location.split(',')[0].strip()
    data['dispozice'] = rooms
    data['užitná_plocha'] = int(meters)

    if 'celková_cena' in data:
        data['cena_czk'] = int(data['celková_cena'].replace('Kč', '').strip()) if 'Kč' in data['celková_cena'] else 'Cena na vyžádání'
        del data['celková_cena']
    if 'energetická_náročnost' in data:
        data['energetická_náročnost'] = data['energetická_náročnost'].split(',')[0]
        s = data['energetická_náročnost'].split(',')[-1].strip()
        m = re.match(r'^(\d+)', s)
        data['energetická_spotřeba_kWh_m2_rok'] = m.group(1) if m else None
    if 'stavba' in data:
        stavba = data['stavba'].split(',')
        
        data['typ_stavby'] = []
        is_first = True
        for part in stavba:
            if 'podlaží' in part.lower() and is_first:
                data['podlaží'] = part.strip()
                is_first = False
                stavba.remove(part)
            elif 'podlaží' in part.lower() and not is_first:
                data['podlaží_další'].append(part.strip())
                stavba.remove(part)
            else:
                data['typ_stavby'].append(part.strip())
        
        del data['stavba']
    if 'infrastruktura' in data:
        infrastruktura = data['infrastruktura'].split(',')
        data['infrastruktura'] = [item.strip() for item in infrastruktura]
    # Handle 'plocha' field - split into separate area types
    if 'plocha' in data:
        plocha_text = data['plocha']
        # Pattern to match area types like "Užitná plocha 44 m²"
        area_pattern = r'([^0-9]+?)\s+(\d+)\s*m²'
        matches = re.findall(area_pattern, plocha_text)
        
        for area_type, area_value in matches:
            area_type = area_type.strip()
            # Convert to snake_case key
            key = area_type.lower().replace(' ', '_')
            data[key] = int(area_value)
        
        # Remove the original 'plocha' field
        del data['plocha']
    if 'ostatní' in data:
        s = str(data['ostatní'])
        start_idx = s.find('K nastěhování')
        if start_idx != -1:
            substr = s[start_idx:]
            m = re.search(r'K nastěhování.*?(\d{4})', substr)
            if m:
                data['k_nastěhování'] = substr[:m.end(1)].strip()
                # remove the matched portion from the original 'ostatní' value
                remove_end = start_idx + m.end(1)
                cleaned = (s[:start_idx] + s[remove_end:]).strip()
                data['ostatní'] = cleaned if cleaned else None
            else:
                # no 4-digit number found after marker — keep the rest of the substring
                data['k_nastěhování'] = substr.strip()
                # remove the marker and the rest from 'ostatní'
                cleaned = s[:start_idx].strip()
                data['ostatní'] = cleaned if cleaned else None
        else:
            data['k_nastěhování'] = None
    else:
        data['k_nastěhování'] = None
        data['ostatní'] = None
    # Handle k_nastěhování - check if it's not None before processing
    nastěhování = data.get('k_nastěhování') or ''
    if 'ihned' in nastěhování.lower():
        data['k_nastěhování'] = 'Ihned'
    elif nastěhování:
        # only keep the date, no k nastěhování text
        m = re.search(r'(\d{1,2}\.\s*\d{1,2}\.\s*\d{4})', nastěhování)
        if m:
            data['k_nastěhování'] = m.group(1)

    return data


preprocessed_data = []
for i in range(len(raw_json)):
    print(f"Preprocessing {i+1}/{len(raw_json)}")
    result = preprocess_data(raw_json[i])
    if result is not None:
        preprocessed_data.append(result)
    else:
        print(f"  -> Skipped listing {i+1} (missing or invalid title)")

print(f"\nSuccessfully preprocessed {len(preprocessed_data)}/{len(raw_json)} listings")

new_json_path = Path(__file__).parent / 'preprocessed_listings.json'

with open(new_json_path, 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)

