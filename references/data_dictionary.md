# Data Dictionary: Red Wine Quality

**Source:** P. Cortez et al. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547-553.

## Variables

| Variable | Type | Unit | Description |
|---|---|---|---|
| `fixed_acidity` | Continuous | g/dm3 | Non-volatile acids (mainly tartaric); contribute to crispness |
| `volatile_acidity` | Continuous | g/dm3 | Acetic acid; at high levels produces a vinegar-like taste |
| `citric_acid` | Continuous | g/dm3 | Adds freshness and flavour |
| `residual_sugar` | Continuous | g/dm3 | Sugar remaining after fermentation |
| `chlorides` | Continuous | g/dm3 | Salt content; affects taste and texture |
| `free_sulfur_dioxide` | Continuous | mg/dm3 | Free SO2; prevents microbial growth and oxidation |
| `total_sulfur_dioxide` | Continuous | mg/dm3 | All SO2 forms (free + bound); detectable at high concentrations |
| `density` | Continuous | g/cm3 | Related to alcohol and sugar content |
| `pH` | Continuous | unitless | Acidity scale; most wines are 3.0-4.0 |
| `sulphates` | Continuous | g/dm3 | Wine additive contributing to SO2 levels |
| `alcohol` | Continuous | % vol | Percentage alcohol by volume |
| **`quality`** | **Ordinal (target)** | **0-10** | **Median sensory score from at least 3 expert sommeliers** |

## Notes

- `free_sulfur_dioxide` is a mathematical subset of `total_sulfur_dioxide` (total = free + bound).
- `quality` is treated as continuous for OLS regression, which is a known simplification over ordinal regression.
- The dataset contains only Portuguese *Vinho Verde* red wines collected between 2004 and 2007.
