# Apartment rental listings — Virginia

`apartments_for_rent_VA.csv` holds 8,284 apartment-for-rent listings in Virginia, scraped
from US rental classified sites (mostly RentDigs.com) between December 2018 and
December 2019. Used in Module 7.3 (multivariate linear regression).

## Source

"Apartment for Rent Classified," UCI Machine Learning Repository, dataset #555 (2019).
https://doi.org/10.24432/C5X623 — licensed CC BY 4.0. No creator is named on the UCI page.

The full dataset has 99,492 US listings (~102 MB, too big to commit). This file keeps only
the rows where `state == "VA"`. Rows and all 22 columns are unchanged; the file was
converted from semicolon-separated cp1252 to comma-separated UTF-8 so that
`pd.read_csv()` works with default settings. Rebuild with `tools/extract_apartments_va.py`.

## Small sample

`apartments_shenandoah_sample.csv` holds 14 listings from the file above (Harrisonburg,
Staunton, Waynesboro), all 22 columns, used for the small worked example in the 7.3
lecture. They were hand-picked, not drawn at random: data errors (a 4,000 sq ft "studio",
a "studio" listed with 2 bedrooms, a 5 BR unit in 1,236 sq ft), repeated postings of the
same unit, and a 4 BR/4 BA student-housing outlier were left out. The listing ids are in
`tools/extract_apartments_va.py`.

## Columns

| Column | Meaning |
|---|---|
| `price` | Monthly rent, USD (the regression target) |
| `square_feet` | Unit size |
| `bedrooms`, `bathrooms` | Counts; 0 bedrooms = studio; half-baths appear as .5 |
| `cityname`, `state` | Location (119 Virginia cities) |
| `latitude`, `longitude` | Location in degrees |
| `amenities` | Comma-separated list, e.g. `Gym,Parking,Pool`; blank when none listed |
| `pets_allowed` | `Cats,Dogs`, `Cats`, `Dogs`, or blank (not stated) |
| `has_photo` | `Yes`, `Thumbnail`, or `No` |
| `fee` | `Yes`/`No` |
| `id`, `category`, `title`, `body`, `address`, `currency`, `price_display`, `price_type`, `source` | Listing details (free text or constant in Virginia) |
| `time` | When the listing was posted, Unix seconds |

## Known quirks

- 6 rows are exact duplicates.
- A few extreme values (a 107 sq ft unit; $7,500/month units in Arlington and Annandale).
- Blank values were written as the text `null` in the original; pandas reads these as missing.
