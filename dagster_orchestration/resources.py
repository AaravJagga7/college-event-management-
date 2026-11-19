"""
================================================================================
Dagster LakeFS Resource Definition
================================================================================
Provides a structured Dagster ConfigurableResource interface for lakeFS data
lake versioning, branching, and commit provenance tracking.
================================================================================
"""

import os
from typing import Dict, Any, Optional
from dagster import ConfigurableResource
from pydantic import Field
from lakefs_versioning import (
    LakeFSManager,
    DEFAULT_ENDPOINT,
    DEFAULT_ACCESS_KEY,
    DEFAULT_SECRET_KEY,
    DEFAULT_REPOSITORY,
    DEFAULT_BRANCH,
    DEFAULT_STORAGE_NAMESPACE
)


class LakeFSResource(ConfigurableResource):
    """
    Dagster resource that manages data lake versioning, metadata provenance,
    and Git-for-data lifecycle actions via lakeFS.
    """
    endpoint: str = Field(default=DEFAULT_ENDPOINT, description="lakeFS API endpoint")
    access_key: str = Field(default=DEFAULT_ACCESS_KEY, description="lakeFS Access Key ID")
    secret_key: str = Field(default=DEFAULT_SECRET_KEY, description="lakeFS Secret Access Key")
    repository: str = Field(default=DEFAULT_REPOSITORY, description="lakeFS repository name")
    branch: str = Field(default=DEFAULT_BRANCH, description="Target lakeFS branch")

    def get_manager(self) -> LakeFSManager:
        """Returns initialized LakeFSManager instance."""
        return LakeFSManager(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            repository=self.repository,
            branch=self.branch
        )

    def get_latest_commit(self) -> Dict[str, Any]:
        """Fetches the latest commit metadata from lakeFS."""
        manager = self.get_manager()
        return manager.get_latest_commit(self.branch)

    def commit_version(
        self,
        raw_path: str = "rvce_8th_mile_data.csv",
        processed_path: str = "features_ready.csv",
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new versioned commit in lakeFS with cryptographic hash provenance."""
        manager = self.get_manager()
        return manager.sync_data(
            raw_path=raw_path,
            processed_path=processed_path,
            commit_msg=message
        )

    def get_lakefs_uri(self, object_path: str) -> str:
        """Constructs canonical lakefs:// URI for a given path."""
        clean_path = object_path.lstrip("/")
        return f"lakefs://{self.repository}/{self.branch}/{clean_path}"
