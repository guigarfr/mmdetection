"""Bounding box generic class."""
from copy import deepcopy

import numpy as np


class BBox2D(object):
    """Two dimensional bounding box."""

    def __init__(self, x):
        """Two dimensional bounding box initialization.

        :param x: Sequence of length 8 representing (x,y,x,y,x,y,x,y)
        :raises:
            :ValueError: If `x` is not of length 8.
            :TypeError:  If `x` is not of type
                         {list, tuple, numpy.ndarray, BBox2D}
        """
        # Copy constructor makes the constructor idempotent
        if isinstance(x, BBox2D):
            x = x.to_numpy()

        elif isinstance(x, (list, tuple)):
            if len(x) != 8:
                raise ValueError(
                    "Invalid input length. Input should have 8 elements.")
            if not all(
                    isinstance(v, (int, float, np.integer, np.float))
                    for v in x):
                raise TypeError('All elements of x should be float or ints')

            x = np.asarray(x)

        elif isinstance(x, np.ndarray):
            if x.ndim >= 2:
                x = x.flatten()

            if x.size != 8:
                raise ValueError(
                    "Invalid input length. Input should have 8 elements.")

            if not all(isinstance(v, (np.integer, np.float)) for v in x):
                raise TypeError('All elements of x should be float or ints')

        else:
            raise TypeError((
                "Expected input to constructor to be a 8 element "
                "list, tuple, numpy ndarray, or BBox2D object."))

        self._x0, self._y0, self._x1, self._y1,\
            self._x2, self._y2, self._x3, self._y3 = x

    def __getstate__(self):
        return dict(
            x0=self.x0,
            y0=self.y0,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            x3=self.x3,
            y3=self.y3,
        )

    def __setstate__(self, state):
        x = [
            state['x0'], state['y0'], state['x1'], state['y1'],
            state['x2'], state['y2'], state['x3'], state['y3']
        ]
        self.__init__(x)

    def __eq__(self, x):
        if not isinstance(x, BBox2D):
            return False
        return all([
            self.x0 == x.x0,
            self.y0 == x.y0,
            self.x1 == x.x1,
            self.y1 == x.y1,
            self.x2 == x.x2,
            self.y2 == x.y2,
            self.x3 == x.x3,
            self.y3 == x.y3,
        ])

    def to_list(self):
        """Bounding box as a `list` of 4 numbers.

        Format depends on ``mode`` flag (default is XYWH).

        :param mode: Mode in which to return the box.
        :type mode: BoxMode2D
        :return: Bounding box as a `list` of 4 numbers.
        """
        return [
            self.x0, self.y0, self.x1, self.y1,
            self.x2, self.y2, self.x3, self.y3
        ]

    def copy(self):
        """Deep copy of this 2D bounding box."""
        return deepcopy(self)

    def to_numpy(self):
        """Bounding box as a numpy vector of length 4.

        Format depends on ``mode`` flag (default is XYWH).

        :param mode: Mode in which to return the box
        :type mode: BoxMode2D
        :return: Bounding box as a numpy vector of length 4
        """
        return np.asarray(self.to_list(), dtype=np.float)

    def __str__(self):
        string = (
            f"BBox2D([{self.x0}, {self.y0}, "
            f"{self.x1}, {self.y1}, {self.x2}, {self.y2}, "
            f"{self.x3}, {self.y3}])"
        )
        return string

    def to_obj(self):
        obj = [
            {
                'x': self.x0,
                'y': self.y0
            },
            {
                'x': self.x1,
                'y': self.y1
            },
            {
                'x': self.x2,
                'y': self.y2
            },
            {
                'x': self.x3,
                'y': self.y3
            }
        ]
        return obj

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def x1(self):
        return self._x1

    @property
    def y1(self):
        return self._y1

    @property
    def x2(self):
        return self._x2

    @property
    def y2(self):
        return self._y2

    @property
    def x3(self):
        return self._x3

    @property
    def y3(self):
        return self._y3
