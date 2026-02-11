import re
from typing import Dict, Optional, Tuple


REQUIRED_FIELDS = ["age", "sex", "bmi", "children", "smoker"]

VALID_SEX = ["male", "female"]
VALID_SMOKER = ["yes", "no"]


class FeatureManager:
    """
    Tracks and validates the required features for insurance pricing.
    """

    def __init__(self):
        self.features = {field: None for field in REQUIRED_FIELDS}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def is_complete(self) -> bool:
        """Return True if all required features are filled."""
        return all(self.features.values())

    def missing_fields(self):
        """Return a list of missing fields."""
        return [f for f, v in self.features.items() if v is None]

    def set_feature(self, field: str, value: str) -> Tuple[bool, Optional[str]]:
        """
        Attempt to set a feature. Returns:
        (success: bool, error_message: Optional[str])
        """
        field = field.lower().strip()

        if field not in REQUIRED_FIELDS:
            return False, f"'{field}' is not a valid field."

        success, parsed_value_or_error = self._validate_and_parse(field, value)

        if not success:
            return False, parsed_value_or_error  # error message

        # Save parsed value
        self.features[field] = parsed_value_or_error
        return True, None

    def get_features(self) -> Dict:
        """Return the completed feature dict (only when is_complete() == True)."""
        return self.features.copy()

    # ---------------------------------------------------------
    # Validation + Parsing
    # ---------------------------------------------------------

    def _validate_and_parse(self, field: str, value: str):
        value = value.strip().lower()

        # AGE
        if field == "age":
            if not value.isdigit():
                return False, "Age must be a whole number."
            age = int(value)
            if age < 0 or age > 120:
                return False, "Age must be between 0 and 120."
            return True, age

        # BMI
        if field == "bmi":
            try:
                bmi = float(value)
            except ValueError:
                return False, "BMI must be a number."
            if bmi < 10 or bmi > 80:
                return False, "BMI must be between 10 and 80."
            return True, bmi

        # CHILDREN
        if field == "children":
            if not value.isdigit():
                return False, "Children must be an integer."
            children = int(value)
            if children < 0 or children > 20:
                return False, "Children must be between 0 and 20."
            return True, children

        # SEX
        if field == "sex":
            if value not in VALID_SEX:
                return False, f"Sex must be one of: {VALID_SEX}"
            return True, value

        # SMOKER
        if field == "smoker":
            if value not in VALID_SMOKER:
                return False, f"Smoker must be one of: {VALID_SMOKER}"
            return True, value


        return False, "Unknown field."
