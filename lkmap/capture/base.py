from __future__ import annotations

from abc import ABC, abstractmethod

from lkmap.models import CaptureRegion, FrameData


class CaptureStrategy(ABC):
    def start(self) -> None:
        """Optional lifecycle hook."""

    def stop(self) -> None:
        """Optional lifecycle hook."""

    @abstractmethod
    def current_frame(self) -> FrameData | None:
        raise NotImplementedError

    def select_region(self, parent=None, initial: CaptureRegion | None = None) -> CaptureRegion | None:
        raise NotImplementedError

