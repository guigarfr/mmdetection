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
            x0=self.x_0,
            y0=self.y_0,
            x1=self.x_1,
            y1=self.y_1,
            x2=self.x_2,
            y2=self.y_2,
            x3=self.x_3,
            y3=self.y_3,
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
            self.x_0 == x.x_0,
            self.y_0 == x.y_0,
            self.x_1 == x.x_1,
            self.y_1 == x.y_1,
            self.x_2 == x.x_2,
            self.y_2 == x.y_2,
            self.x_3 == x.x_3,
            self.y_3 == x.y_3,
        ])

    def to_list(self):
        """Bounding box as a `list` of 4 numbers.

        Format depends on ``mode`` flag (default is XYWH).

        :param mode: Mode in which to return the box.
        :type mode: BoxMode2D
        :return: Bounding box as a `list` of 4 numbers.
        """
        return [
            self.x_0, self.y_0, self.x_1, self.y_1,
            self.x_2, self.y_2, self.x_3, self.y_3
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
            f"BBox2D([{self.x_0}, {self.y_0}, "
            f"{self.x_1}, {self.y_1}, {self.x_2}, {self.y_2}, "
            f"{self.x_3}, {self.y_3}])"
        )
        return string

    def to_obj(self):
        obj = [
            {
                'x': self.x_0,
                'y': self.y_0
            },
            {
                'x': self.x_1,
                'y': self.y_1
            },
            {
                'x': self.x_2,
                'y': self.y_2
            },
            {
                'x': self.x_3,
                'y': self.y_3
            }
        ]
        return obj

    @property
    def x_0(self):
        return self._x0

    @property
    def y_0(self):
        return self._y0

    @property
    def x_1(self):
        return self._x1

    @property
    def y_1(self):
        return self._y1

    @property
    def x_2(self):
        return self._x2

    @property
    def y_2(self):
        return self._y2

    @property
    def x_3(self):
        return self._x3

    @property
    def y_3(self):
        return self._y3
