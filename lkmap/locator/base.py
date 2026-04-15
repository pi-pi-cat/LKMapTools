from __future__ import annotations

from abc import ABC, abstractmethod

from lkmap.models import FrameData, LocationResult


class LocatorStrategy(ABC):
    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def locate(self, frame: FrameData) -> LocationResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

