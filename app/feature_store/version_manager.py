import logging
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from .config import FeatureStoreConfig, REDIS_KEY_PATTERNS

logger = logging.getLogger(__name__)

class FeatureVersionManager:
    def __init__(self, config: FeatureStoreConfig):
        self.config = config

        self.version_counter = 0
        self.version_history = {}

        logger.info("Feature Version Manager initialized")

    def create_version(
        self,
        entity_id: str,
        features: Dict[str, Any],
        version_tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:

        try:
            version_id = version_tag or self._generate_version_id()

            version_record = {
                "version_id": version_id,
                "entity_id": entity_id,
                "features": features,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "feature_count": len(features),
                "feature_names": list(features.keys())
            }

            if entity_id not in self.version_history:
                self.version_history[entity_id] = []

            self.version_history[entity_id].append(version_record)

            if len(self.version_history[entity_id]) > self.config.max_versions:
                self.version_history[entity_id] = self.version_history[entity_id][-self.config.max_versions:]

            logger.debug(f"Created version {version_id} for entity {entity_id}")
            return version_id

        except Exception as e:
            logger.error(f"Error creating version: {e}")
            return ""

    def get_version(
        self,
        entity_id: str,
        version_id: str
    ) -> Optional[Dict[str, Any]]:

        try:
            if entity_id not in self.version_history:
                return None

            for version_record in self.version_history[entity_id]:
                if version_record["version_id"] == version_id:
                    return version_record

            return None

        except Exception as e:
            logger.error(f"Error retrieving version: {e}")
            return None

    def list_versions(self, entity_id: str) -> List[str]:

        try:
            if entity_id not in self.version_history:
                return []

            versions = [
                record["version_id"]
                for record in reversed(self.version_history[entity_id])
            ]

            return versions

        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []

    def get_latest_version(self, entity_id: str) -> Optional[Dict[str, Any]]:

        try:
            if entity_id not in self.version_history or not self.version_history[entity_id]:
                return None

            return self.version_history[entity_id][-1]

        except Exception as e:
            logger.error(f"Error getting latest version: {e}")
            return None

    def compare_versions(
        self,
        entity_id: str,
        version_1: str,
        version_2: str
    ) -> Dict[str, Any]:

        try:
            v1_record = self.get_version(entity_id, version_1)
            v2_record = self.get_version(entity_id, version_2)

            if not v1_record or not v2_record:
                return {
                    "error": "One or both versions not found",
                    "version_1_found": v1_record is not None,
                    "version_2_found": v2_record is not None
                }

            v1_features = v1_record["features"]
            v2_features = v2_record["features"]

            added_features = set(v2_features.keys()) - set(v1_features.keys())
            removed_features = set(v1_features.keys()) - set(v2_features.keys())
            common_features = set(v1_features.keys()) & set(v2_features.keys())

            modified_features = {}
            for feature_name in common_features:
                if v1_features[feature_name] != v2_features[feature_name]:
                    modified_features[feature_name] = {
                        "old_value": v1_features[feature_name],
                        "new_value": v2_features[feature_name]
                    }

            comparison = {
                "entity_id": entity_id,
                "version_1": {
                    "id": version_1,
                    "created_at": v1_record["created_at"],
                    "feature_count": len(v1_features)
                },
                "version_2": {
                    "id": version_2,
                    "created_at": v2_record["created_at"],
                    "feature_count": len(v2_features)
                },
                "differences": {
                    "added_features": list(added_features),
                    "removed_features": list(removed_features),
                    "modified_features": modified_features
                },
                "summary": {
                    "total_changes": len(added_features) + len(removed_features) + len(modified_features),
                    "has_changes": bool(added_features or removed_features or modified_features)
                }
            }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {"error": str(e)}

    def get_feature_lineage(
        self,
        entity_id: str,
        feature_name: str
    ) -> Dict[str, Any]:

        try:
            if entity_id not in self.version_history:
                return {"error": "No version history found for entity"}

            lineage = {
                "entity_id": entity_id,
                "feature_name": feature_name,
                "history": [],
                "first_seen": None,
                "last_updated": None,
                "version_count": 0
            }

            for version_record in self.version_history[entity_id]:
                if feature_name in version_record["features"]:
                    history_entry = {
                        "version_id": version_record["version_id"],
                        "value": version_record["features"][feature_name],
                        "created_at": version_record["created_at"],
                        "metadata": version_record.get("metadata", {})
                    }

                    lineage["history"].append(history_entry)

                    if lineage["first_seen"] is None:
                        lineage["first_seen"] = version_record["created_at"]

                    lineage["last_updated"] = version_record["created_at"]
                    lineage["version_count"] += 1

            return lineage

        except Exception as e:
            logger.error(f"Error getting feature lineage: {e}")
            return {"error": str(e)}

    def rollback_to_version(
        self,
        entity_id: str,
        target_version: str
    ) -> Optional[Dict[str, Any]]:

        try:
            target_record = self.get_version(entity_id, target_version)

            if not target_record:
                logger.error(f"Target version {target_version} not found")
                return None

            rollback_features = target_record["features"].copy()

            rollback_metadata = {
                "rollback": True,
                "rollback_from_version": self.get_latest_version(entity_id)["version_id"],
                "rollback_to_version": target_version,
                "rollback_timestamp": datetime.now().isoformat()
            }

            new_version_id = self.create_version(
                entity_id=entity_id,
                features=rollback_features,
                metadata=rollback_metadata
            )

            logger.info(f"Rolled back entity {entity_id} to version {target_version}")

            return {
                "success": True,
                "new_version_id": new_version_id,
                "rollback_metadata": rollback_metadata
            }

        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return None

    def cleanup_old_versions(self, max_age_days: int = 30) -> int:

        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cleanup_count = 0

            for entity_id in list(self.version_history.keys()):
                versions = self.version_history[entity_id]

                filtered_versions = [
                    v for v in versions
                    if datetime.fromisoformat(v["created_at"]) > cutoff_date
                ]

                if filtered_versions:
                    self.version_history[entity_id] = filtered_versions
                else:
                    self.version_history[entity_id] = versions[-1:] if versions else []

                cleanup_count += len(versions) - len(self.version_history[entity_id])

            logger.info(f"Cleaned up {cleanup_count} old versions")
            return cleanup_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def get_version_statistics(self) -> Dict[str, Any]:

        try:
            total_entities = len(self.version_history)
            total_versions = sum(len(versions) for versions in self.version_history.values())

            if total_entities == 0:
                return {
                    "total_entities": 0,
                    "total_versions": 0,
                    "average_versions_per_entity": 0.0
                }

            average_versions = total_versions / total_entities

            entity_version_counts = [
                (entity_id, len(versions))
                for entity_id, versions in self.version_history.items()
            ]
            entity_version_counts.sort(key=lambda x: x[1], reverse=True)

            return {
                "total_entities": total_entities,
                "total_versions": total_versions,
                "average_versions_per_entity": round(average_versions, 2),
                "entities_with_most_versions": entity_version_counts[:5]
            }

        except Exception as e:
            logger.error(f"Error getting version statistics: {e}")
            return {}

    def export_version_history(self, entity_id: Optional[str] = None) -> Dict[str, Any]:

        try:
            if entity_id:
                if entity_id in self.version_history:
                    export_data = {entity_id: self.version_history[entity_id]}
                else:
                    export_data = {}
            else:
                export_data = self.version_history.copy()

            return {
                "version_history": export_data,
                "exported_at": datetime.now().isoformat(),
                "config": {
                    "max_versions": self.config.max_versions,
                    "versioning_enabled": self.config.enable_versioning
                }
            }

        except Exception as e:
            logger.error(f"Error exporting version history: {e}")
            return {}

    def import_version_history(self, import_data: Dict[str, Any]) -> bool:

        try:
            if "version_history" in import_data:
                imported_history = import_data["version_history"]

                for entity_id, versions in imported_history.items():
                    if entity_id not in self.version_history:
                        self.version_history[entity_id] = []

                    self.version_history[entity_id].extend(versions)

                    if len(self.version_history[entity_id]) > self.config.max_versions:
                        self.version_history[entity_id] = self.version_history[entity_id][-self.config.max_versions:]

                logger.info("Successfully imported version history")
                return True

            return False

        except Exception as e:
            logger.error(f"Error importing version history: {e}")
            return False

    def _generate_version_id(self) -> str:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"v_{timestamp}_{unique_id}"
