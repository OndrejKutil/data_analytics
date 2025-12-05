# Feature Extraction: What and Why

This document lists the features extracted from raw listings and why they matter for pricing/valuation.

## Source Files

- Scraper: `src/scraper/scraper.py`
- Preprocessor: `src/preprocessing/preprocessing.py`

## Core Targets

- `price_czk`: Observed total price; used for supervised training and sanity checks.
- Optional derived target: price per m² (can be derived as `price_czk / usable_area` when available).

## Location Features

- `city`, `district`, `street`, `zip_code`: Location strongly drives price via proximity to jobs, services, and demand pockets.
- `city_part`/`ward`/`quarter` (when available): Finer-grained neighborhood signals.
- Latitude/Longitude (from scraper JSON): Enables distance features (e.g., to center/POIs) and spatial smoothing.

## Structural / Configuration

- `layout` (e.g., 2+kk, 3+1): Impacts utility and household fit.
- `floor`, `total_floors`: Views, noise, and elevator need; higher floors can command premium in some markets.
- `building_type` (Panel/Brick/etc.): Construction impacts insulation, aesthetics, and buyer preferences.
- `condition` (e.g., New build, After renovation): Directly affects expected renovation cost and buyer effort.
- `ownership`: Legal form can change financing options and buyer pool.

## Areas (Size)

- `usable_area`, `floor_area`: Primary drivers of price; often monotonic relationships with diminishing returns.
- `balcony_area`, `terrace_area`, `loggia_area`, `cellar_area`, `garden_area`: Outdoor/storage space adds utility and desirability.

## Amenities (Structured)

- `has_balcony`, `has_terrace`, `has_loggia`, `has_cellar`: Outdoor/auxiliary space signals.
- `has_garage`, `has_parking`: Parking availability is a strong driver in dense areas.
- `has_elevator`: Critical for upper floors; broadens buyer pool.
- `furnished`: Potential short-term rental appeal; can be a small premium.

## Energy / Utilities

- `energy_rating`, `energy_performance_certificate`, `low_energy`: Efficiency drives operating costs and modern standards compliance.
- Heating types/sources/elements: Comfort and operating cost signals (from scraper param sets).
- Water/sewage/gas/electricity types: Availability/quality signals (rural vs urban differences).

## Transport / Surroundings

- `transport_types`, `road_types`: Connectivity signals.
- `object_location`, `surroundings_type`: Setting (e.g., city center, residential, quiet zone).
- POI counts (nearby transport/doctors/grocery/leisure/restaurants/schools): Service density proxies that affect valuation.

## Text-Derived Features (Description)

From `_extract_features_from_description`:

- Access: `desc_has_metro`, `desc_has_tram`, `desc_has_bus` – convenient transport often boosts prices.
- Green/recreation: `desc_has_park` – impacts quality of life.
- Family orientation: `desc_has_school` – proximity to schools increases demand.
- Lifestyle: `desc_has_shopping`, `desc_is_quiet`, `desc_is_sunny` – comfort and convenience descriptors.
- Equipment/comfort: `desc_has_ac`, `desc_has_fireplace`, `desc_has_floor_heating` – signals upgrades and comfort.
- Condition: `desc_is_renovated`, `desc_is_new_building` – consistency checks with structured fields.

## Consolidated Flags

- `is_renovated`: True if condition indicates renovation or description claims it.
- `is_new_building`: True if condition says "Novostavba" or description signals new build.

## Notes and Future Enhancements

- Categorical encoding: Many string features need encoding (target/ordered/one-hot) in modeling.
- Text: Current approach is keyword flags; consider TF‑IDF or small embeddings to capture richer semantics.
- Deduplication: Use `listing_id` to merge multiple scrapes of the same listing.
- Quality: Normalize units (m², floors), and validate numeric ranges to catch outliers.
