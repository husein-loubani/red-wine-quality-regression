"""
config.py
Global constants and style settings for the Red Wine Quality Regression project.
All notebooks and src modules should import from here instead of defining constants locally.
"""

# Reproducibility
RANDOM_SEED: int = 42
ALPHA: float = 0.05
TEST_SIZE: float = 0.20

# Visual style
ACCENT_COLOR: str = "#c0392b"   # deep red, thematic for wine
BLUE_COLOR: str = "#2980b9"
PALETTE: str = "muted"
FIGURE_DPI: int = 120

# Feature lists (free_sulfur_dioxide is excluded because it is conceptually
# contained within total_sulfur_dioxide; keeping both would create interpretive
# overlap in the explanatory model)
SKEW_CANDIDATES: list[str] = [
    "residual_sugar",
    "chlorides",
    "total_sulfur_dioxide",
    "sulphates",
]

LOG_FEATURES: list[str] = [f"log_{col}" for col in SKEW_CANDIDATES]

# Full model predictors (density excluded for multicollinearity)
FULL_PREDICTORS: list[str] = [
    "volatile_acidity",
    "log_chlorides",
    "log_total_sulfur_dioxide",
    "log_sulphates",
    "alcohol",
    "fixed_acidity",
    "pH",
    "citric_acid",
    "log_residual_sugar",
]

# Reduced model predictors (non-significant terms removed for parsimony)
REDUCED_PREDICTORS: list[str] = [
    "volatile_acidity",
    "log_chlorides",
    "log_total_sulfur_dioxide",
    "log_sulphates",
    "alcohol",
    "pH",
]

FULL_OLS_FORMULA: str = (
    "quality ~ volatile_acidity"
    " + log_chlorides"
    " + log_total_sulfur_dioxide"
    " + log_sulphates"
    " + alcohol"
    " + fixed_acidity"
    " + pH"
    " + citric_acid"
    " + log_residual_sugar"
)

# Stepwise backward elimination formulas
# Step 1: remove log_residual_sugar (highest p-value in full model)
STEP1_OLS_FORMULA: str = (
    "quality ~ volatile_acidity"
    " + log_chlorides"
    " + log_total_sulfur_dioxide"
    " + log_sulphates"
    " + alcohol"
    " + fixed_acidity"
    " + pH"
    " + citric_acid"
)

# Step 2: remove citric_acid (highest p-value in step-1 model)
STEP2_OLS_FORMULA: str = (
    "quality ~ volatile_acidity"
    " + log_chlorides"
    " + log_total_sulfur_dioxide"
    " + log_sulphates"
    " + alcohol"
    " + fixed_acidity"
    " + pH"
)

# Step 3 (reduced): remove fixed_acidity (highest p-value in step-2 model)
REDUCED_OLS_FORMULA: str = (
    "quality ~ volatile_acidity"
    " + log_chlorides"
    " + log_total_sulfur_dioxide"
    " + log_sulphates"
    " + alcohol"
    " + pH"
)

STEP_FORMULAS: list[tuple[str, str]] = [
    ("Full (9 predictors)", FULL_OLS_FORMULA),
    ("Step 1: drop log_residual_sugar", STEP1_OLS_FORMULA),
    ("Step 2: drop citric_acid", STEP2_OLS_FORMULA),
    ("Step 3: drop fixed_acidity (reduced)", REDUCED_OLS_FORMULA),
]

# Column rename map (raw CSV names to Python-safe names)
COLUMN_RENAME: dict[str, str] = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
}
