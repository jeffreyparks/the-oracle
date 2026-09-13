"""Domain packs: shared objective library plus per-domain manifests."""

from the_oracle.domains.errors import ObjectiveNotFoundError, PackValidationError
from the_oracle.domains.registry import (
    list_domains,
    load_domain,
    oracle_home,
    validate_domain,
)
from the_oracle.domains.schema import (
    Domain,
    Edge,
    Misconception,
    Module,
    Objective,
    ObjectiveRef,
)

__all__ = [
    "Domain",
    "Edge",
    "Misconception",
    "Module",
    "Objective",
    "ObjectiveNotFoundError",
    "ObjectiveRef",
    "PackValidationError",
    "list_domains",
    "load_domain",
    "oracle_home",
    "validate_domain",
]
