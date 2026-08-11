"""Geometric primitives for OCR bounding regions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """A two-dimensional point in image coordinate space.

    Attributes:
        x: Horizontal coordinate in pixels.
        y: Vertical coordinate in pixels.
    """

    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned rectangular bounding box.

    Attributes:
        x_min: Left edge x-coordinate in pixels.
        y_min: Top edge y-coordinate in pixels.
        x_max: Right edge x-coordinate in pixels.
        y_max: Bottom edge y-coordinate in pixels.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Compute the box width in pixels.

        Returns:
            The horizontal span of the bounding box.
        """
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Compute the box height in pixels.

        Returns:
            The vertical span of the bounding box.
        """
        return self.y_max - self.y_min

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return coordinates as a flat tuple.

        Returns:
            A tuple of ``(x_min, y_min, x_max, y_max)``.
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass(frozen=True)
class Polygon:
    """A closed polygon defined by an ordered vertex sequence.

    Attributes:
        points: Polygon vertices in consecutive order.
    """

    points: tuple[Point, ...]

    @property
    def point_count(self) -> int:
        """Return the number of polygon vertices.

        Returns:
            The vertex count.
        """
        return len(self.points)
