"""Pluggable indicator / interaction registry."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Type

from features.exceptions import RegistryError
from features.indicators.base_indicator import BaseIndicator

logger = logging.getLogger(__name__)


class IndicatorRegistry:
    """Maps registry keys → indicator classes. Pipeline never hardcodes classes."""

    _registry: dict[str, Type[BaseIndicator]] = {}

    @classmethod
    def register(cls, indicator_cls: Type[BaseIndicator]) -> Type[BaseIndicator]:
        key = indicator_cls({}).name()
        if key in cls._registry and cls._registry[key] is not indicator_cls:
            raise RegistryError(f"duplicate indicator registration: {key!r}")
        cls._registry[key] = indicator_cls
        logger.debug("Registered indicator | key=%s cls=%s", key, indicator_cls.__name__)
        return indicator_cls

    @classmethod
    def create(cls, key: str, params: dict | None = None) -> BaseIndicator:
        if key not in cls._registry:
            raise RegistryError(
                f"unknown indicator {key!r}; known={sorted(cls._registry)}"
            )
        return cls._registry[key](params)

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def discover(cls) -> None:
        """Import indicator + interaction packages so @register side-effects run."""
        for package_name in ("features.indicators", "features.interactions"):
            try:
                package = importlib.import_module(package_name)
            except ModuleNotFoundError:
                continue
            if not hasattr(package, "__path__"):
                continue
            for mod in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                name = mod.name.rsplit(".", 1)[-1]
                if name.startswith("_") or name in {"base_indicator", "base_interaction"}:
                    continue
                importlib.import_module(mod.name)
        logger.info("IndicatorRegistry discover complete | keys=%s", cls.keys())
