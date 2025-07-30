import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json

from .config import (
    FeatureStoreConfig,
    PRODUCT_FEATURES,
    FeatureType,
    FeatureGroup,
    COMPUTED_FEATURES
)

logger = logging.getLogger(__name__)

class FeatureRegistry:
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.feature_schemas = PRODUCT_FEATURES.copy()
        self.computed_features = COMPUTED_FEATURES.copy()

        self.feature_stats = {}

        logger.info("Feature Registry initialized with {} features".format(
            len(self.feature_schemas)
        ))

    def register_feature(
        self,
        feature_name: str,
        feature_type: FeatureType,
        feature_group: FeatureGroup,
        description: str,
        nullable: bool = True,
        constraints: Optional[Dict[str, Any]] = None
    ) -> bool:

        try:
            if feature_name in self.feature_schemas:
                logger.warning(f"Feature {feature_name} already registered, updating...")

            feature_schema = {
                "type": feature_type,
                "group": feature_group,
                "description": description,
                "nullable": nullable,
                "registered_at": datetime.now().isoformat(),
                "constraints": constraints or {}
            }

            self.feature_schemas[feature_name] = feature_schema

            logger.info(f"Registered feature: {feature_name}")
            return True

        except Exception as e:
            logger.error(f"Error registering feature {feature_name}: {e}")
            return False

    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate features against registered schemas.

        Args:
            features: Dictionary of feature_name -> feature_value

        Returns:
            Dictionary with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "validated_features": Dict[str, Any]
            }
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "validated_features": {}
        }

        for feature_name, feature_value in features.items():
            try:
                if feature_name.startswith("_"):
                    validation_result["validated_features"][feature_name] = feature_value
                    continue

                if feature_name not in self.feature_schemas:
                    validation_result["warnings"].append(
                        f"Feature '{feature_name}' not registered in schema"
                    )
                    validation_result["validated_features"][feature_name] = feature_value
                    continue

                schema = self.feature_schemas[feature_name]

                feature_validation = self._validate_single_feature(
                    feature_name, feature_value, schema
                )

                if feature_validation["valid"]:
                    validation_result["validated_features"][feature_name] = feature_validation["value"]
                else:
                    validation_result["errors"].extend(feature_validation["errors"])
                    validation_result["valid"] = False

            except Exception as e:
                error_msg = f"Validation error for feature '{feature_name}': {e}"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False

        return validation_result

    def _validate_single_feature(
        self,
        feature_name: str,
        feature_value: Any,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:

        result = {
            "valid": True,
            "errors": [],
            "value": feature_value
        }

        try:
            if feature_value is None:
                if not schema.get("nullable", True):
                    result["errors"].append(f"Feature '{feature_name}' cannot be null")
                    result["valid"] = False
                return result

            feature_type = schema["type"]

            if feature_type == FeatureType.NUMERIC:
                if not isinstance(feature_value, (int, float)):
                    result["errors"].append(
                        f"Feature '{feature_name}' must be numeric, got {type(feature_value)}"
                    )
                    result["valid"] = False

            elif feature_type == FeatureType.CATEGORICAL:
                if not isinstance(feature_value, str):
                    result["errors"].append(
                        f"Feature '{feature_name}' must be string, got {type(feature_value)}"
                    )
                    result["valid"] = False

            elif feature_type == FeatureType.BOOLEAN:
                if not isinstance(feature_value, bool):
                    result["errors"].append(
                        f"Feature '{feature_name}' must be boolean, got {type(feature_value)}"
                    )
                    result["valid"] = False

            elif feature_type in [FeatureType.TEXT_EMBEDDING, FeatureType.IMAGE_EMBEDDING]:
                if isinstance(feature_value, list):
                    expected_dim = schema.get("dimension")
                    if expected_dim and len(feature_value) != expected_dim:
                        result["errors"].append(
                            f"Feature '{feature_name}' expected dimension {expected_dim}, "
                            f"got {len(feature_value)}"
                        )
                        result["valid"] = False
                else:
                    result["errors"].append(
                        f"Feature '{feature_name}' must be a list/array, got {type(feature_value)}"
                    )
                    result["valid"] = False

            constraints = schema.get("constraints", {})
            constraint_validation = self._validate_constraints(
                feature_name, feature_value, constraints
            )

            if not constraint_validation["valid"]:
                result["errors"].extend(constraint_validation["errors"])
                result["valid"] = False

        except Exception as e:
            result["errors"].append(f"Validation exception for '{feature_name}': {e}")
            result["valid"] = False

        return result

    def _validate_constraints(
        self,
        feature_name: str,
        feature_value: Any,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:

        result = {
            "valid": True,
            "errors": []
        }

        try:
            if "min_value" in constraints:
                if isinstance(feature_value, (int, float)):
                    if feature_value < constraints["min_value"]:
                        result["errors"].append(
                            f"Feature '{feature_name}' value {feature_value} "
                            f"below minimum {constraints['min_value']}"
                        )
                        result["valid"] = False

            if "max_value" in constraints:
                if isinstance(feature_value, (int, float)):
                    if feature_value > constraints["max_value"]:
                        result["errors"].append(
                            f"Feature '{feature_name}' value {feature_value} "
                            f"above maximum {constraints['max_value']}"
                        )
                        result["valid"] = False

            if "allowed_values" in constraints:
                if feature_value not in constraints["allowed_values"]:
                    result["errors"].append(
                        f"Feature '{feature_name}' value '{feature_value}' "
                        f"not in allowed values: {constraints['allowed_values']}"
                    )
                    result["valid"] = False

            if "min_length" in constraints:
                if isinstance(feature_value, str):
                    if len(feature_value) < constraints["min_length"]:
                        result["errors"].append(
                            f"Feature '{feature_name}' string length {len(feature_value)} "
                            f"below minimum {constraints['min_length']}"
                        )
                        result["valid"] = False

            if "max_length" in constraints:
                if isinstance(feature_value, str):
                    if len(feature_value) > constraints["max_length"]:
                        result["errors"].append(
                            f"Feature '{feature_name}' string length {len(feature_value)} "
                            f"above maximum {constraints['max_length']}"
                        )
                        result["valid"] = False

        except Exception as e:
            result["errors"].append(f"Constraint validation error: {e}")
            result["valid"] = False

        return result

    def get_feature_schema(self, feature_name: str) -> Optional[Dict[str, Any]]:

        return self.feature_schemas.get(feature_name)

    def list_features(
        self,
        feature_group: Optional[FeatureGroup] = None,
        feature_type: Optional[FeatureType] = None
    ) -> List[str]:

        features = []

        for feature_name, schema in self.feature_schemas.items():
            if feature_group and schema.get("group") != feature_group:
                continue

            if feature_type and schema.get("type") != feature_type:
                continue

            features.append(feature_name)

        return features

    def get_feature_groups(self) -> Dict[FeatureGroup, List[str]]:

        groups = {}

        for feature_name, schema in self.feature_schemas.items():
            group = schema.get("group")
            if group:
                if group not in groups:
                    groups[group] = []
                groups[group].append(feature_name)

        return groups

    def get_feature_types(self) -> Dict[FeatureType, List[str]]:

        types = {}

        for feature_name, schema in self.feature_schemas.items():
            feature_type = schema.get("type")
            if feature_type:
                if feature_type not in types:
                    types[feature_type] = []
                types[feature_type].append(feature_name)

        return types

    def update_feature_statistics(
        self,
        feature_name: str,
        stats: Dict[str, Any]
    ) -> None:

        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {}

        self.feature_stats[feature_name].update(stats)
        self.feature_stats[feature_name]["last_updated"] = datetime.now().isoformat()

    def get_feature_statistics(self, feature_name: str) -> Dict[str, Any]:

        return self.feature_stats.get(feature_name, {})

    def export_schema(self) -> Dict[str, Any]:

        return {
            "features": self.feature_schemas,
            "computed_features": self.computed_features,
            "statistics": self.feature_stats,
            "exported_at": datetime.now().isoformat()
        }

    def import_schema(self, schema_data: Dict[str, Any]) -> bool:

        try:
            if "features" in schema_data:
                self.feature_schemas.update(schema_data["features"])

            if "computed_features" in schema_data:
                self.computed_features.update(schema_data["computed_features"])

            if "statistics" in schema_data:
                self.feature_stats.update(schema_data["statistics"])

            logger.info("Successfully imported feature schema")
            return True

        except Exception as e:
            logger.error(f"Error importing schema: {e}")
            return False

    def validate_feature_dependencies(self, features: Dict[str, Any]) -> Dict[str, Any]:

        result = {
            "valid": True,
            "errors": [],
            "missing_dependencies": {}
        }

        for feature_name in features:
            if feature_name in self.computed_features:
                computed_config = self.computed_features[feature_name]
                dependencies = computed_config.get("depends_on", [])

                missing = [dep for dep in dependencies if dep not in features]

                if missing:
                    result["missing_dependencies"][feature_name] = missing
                    result["errors"].append(
                        f"Computed feature '{feature_name}' missing dependencies: {missing}"
                    )
                    result["valid"] = False

        return result

    def get_feature_documentation(self) -> str:

        doc_lines = ["# Feature Store Documentation\n"]

        feature_types = self.get_feature_types()

        for feature_type, feature_names in feature_types.items():
            doc_lines.append(f"## {feature_type.value.title()} Features\n")

            for feature_name in feature_names:
                schema = self.feature_schemas[feature_name]
                doc_lines.append(f"### {feature_name}")
                doc_lines.append(f"- **Description**: {schema.get('description', 'No description')}")
                doc_lines.append(f"- **Type**: {schema.get('type', {}).value if hasattr(schema.get('type'), 'value') else schema.get('type')}")
                doc_lines.append(f"- **Group**: {schema.get('group', {}).value if hasattr(schema.get('group'), 'value') else schema.get('group')}")
                doc_lines.append(f"- **Nullable**: {schema.get('nullable', True)}")

                if schema.get("constraints"):
                    doc_lines.append(f"- **Constraints**: {schema['constraints']}")

                doc_lines.append("")

        return "\n".join(doc_lines)
