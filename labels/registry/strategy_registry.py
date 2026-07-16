"""Strategy registry — pipeline never hardcodes strategy classes."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Type

from labels.exceptions import RegistryError
from labels.strategies.base_strategy import BaseLabelStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    _registry: dict[str, Type[BaseLabelStrategy]] = {}

    @classmethod
    def register(cls, strategy_cls: Type[BaseLabelStrategy]) -> Type[BaseLabelStrategy]:
        key = strategy_cls({}).name()
        if key in cls._registry and cls._registry[key] is not strategy_cls:
            raise RegistryError(f"duplicate strategy registration: {key!r}")
        cls._registry[key] = strategy_cls
        logger.debug("Registered strategy | key=%s cls=%s", key, strategy_cls.__name__)
        return strategy_cls

    @classmethod
    def create(cls, key: str, params: dict | None = None) -> BaseLabelStrategy:
        if key not in cls._registry:
            raise RegistryError(f"unknown strategy {key!r}; known={sorted(cls._registry)}")
        return cls._registry[key](params)

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def discover(cls) -> None:
        package = importlib.import_module("labels.strategies")
        for mod in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            name = mod.name.rsplit(".", 1)[-1]
            if name.startswith("_") or name == "base_strategy":
                continue
            importlib.import_module(mod.name)
        logger.info("StrategyRegistry discover complete | keys=%s", cls.keys())
