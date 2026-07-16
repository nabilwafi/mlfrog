"""Trainer registry — TrainingPipeline never hardcodes algorithm classes."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Type

from models.exceptions import RegistryError
from models.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class TrainerRegistry:
    _registry: dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, trainer_cls: Type[BaseTrainer]) -> Type[BaseTrainer]:
        key = trainer_cls({}).name()
        existing = cls._registry.get(key)
        if existing is not None and existing is not trainer_cls and existing.__name__ != trainer_cls.__name__:
            raise RegistryError(f"duplicate trainer registration: {key!r}")
        cls._registry[key] = trainer_cls
        logger.debug("Registered trainer | key=%s cls=%s", key, trainer_cls.__name__)
        return trainer_cls

    @classmethod
    def create(cls, key: str, params: dict | None = None) -> BaseTrainer:
        if key not in cls._registry:
            raise RegistryError(f"unknown trainer {key!r}; known={sorted(cls._registry)}")
        return cls._registry[key](params)

    @classmethod
    def keys(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def discover(cls) -> None:
        import sys

        package = importlib.import_module("models.trainers")
        for mod in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            name = mod.name.rsplit(".", 1)[-1]
            if name.startswith("_") or name == "base_trainer":
                continue
            # reload so clear()+discover() re-binds after tests wipe the map
            if mod.name in sys.modules:
                importlib.reload(sys.modules[mod.name])
            else:
                importlib.import_module(mod.name)
        logger.info("TrainerRegistry discover complete | keys=%s", cls.keys())
