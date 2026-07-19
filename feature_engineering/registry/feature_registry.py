"""Feature builder registry — service never hardcodes builder classes."""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from typing import Type

from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.exceptions import RegistryError

logger = logging.getLogger(__name__)


class FeatureRegistry:
    _registry: dict[str, Type[BaseFeatureBuilder]] = {}

    @classmethod
    def register(cls, builder_cls: Type[BaseFeatureBuilder]) -> Type[BaseFeatureBuilder]:
        key = builder_cls({}).name()
        existing = cls._registry.get(key)
        if (
            existing is not None
            and existing is not builder_cls
            and existing.__name__ != builder_cls.__name__
        ):
            raise RegistryError(f"duplicate feature builder: {key!r}")
        cls._registry[key] = builder_cls
        logger.debug("Registered feature builder | key=%s", key)
        return builder_cls

    @classmethod
    def create(cls, key: str, params: dict | None = None) -> BaseFeatureBuilder:
        if key not in cls._registry:
            raise RegistryError(f"unknown builder {key!r}; known={sorted(cls._registry)}")
        return cls._registry[key](params)

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def discover(cls) -> None:
        package = importlib.import_module("feature_engineering.builders")
        for mod in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            name = mod.name.rsplit(".", 1)[-1]
            if name.startswith("_") or name == "base_builder":
                continue
            if mod.name in sys.modules:
                importlib.reload(sys.modules[mod.name])
            else:
                importlib.import_module(mod.name)
        logger.info("FeatureRegistry discover complete | keys=%s", cls.keys())
