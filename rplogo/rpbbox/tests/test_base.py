import pickle
import unittest
from unittest import mock

import numpy as np

from rplogo import rpbbox


class BBox2DTests(unittest.TestCase):
    """Two dimensional bounding box tests."""

    def setUp(self) -> None:
        self.expected_x0 = 0
        self.expected_y0 = 0
        self.expected_x1 = 1
        self.expected_y1 = 1
        self.expected_x2 = 2
        self.expected_y2 = 2
        self.expected_x3 = 3
        self.expected_y3 = 3

        self.xy_values = (
            self.expected_x0,
            self.expected_y0,
            self.expected_x1,
            self.expected_y1,
            self.expected_x2,
            self.expected_y2,
            self.expected_x3,
            self.expected_y3,
        )

    def assertIsExpectedBox(self, box):
        self.assertEqual(self.expected_x0, box.x_0)
        self.assertEqual(self.expected_y0, box.y_0)
        self.assertEqual(self.expected_x1, box.x_1)
        self.assertEqual(self.expected_y1, box.y_1)
        self.assertEqual(self.expected_x2, box.x_2)
        self.assertEqual(self.expected_y2, box.y_2)
        self.assertEqual(self.expected_x3, box.x_3)
        self.assertEqual(self.expected_y3, box.y_3)

    def test_init__bbox(self):
        box1 = rpbbox.BBox2D(self.xy_values)
        box2 = rpbbox.BBox2D(box1)
        self.assertEqual(box1, box2)
        self.assertIsNot(box1, box2)

    def test_init__list(self):
        box = rpbbox.BBox2D(list(self.xy_values))
        self.assertIsExpectedBox(box)

    def test_init__list__invalid_size(self):
        with self.assertRaises(ValueError) as ctx:
            rpbbox.BBox2D(list(self.xy_values)[:3])
        self.assertEqual(
            ctx.exception.args[0],
            'Invalid input length. Input should have 8 elements.'
        )

    def test_init__numpy(self):
        box = rpbbox.BBox2D(np.asarray(self.xy_values))
        self.assertIsExpectedBox(box)

    def test_init__numpy__invalid_size(self):
        with self.assertRaises(ValueError) as ctx:
            rpbbox.BBox2D(
                np.asarray(list(self.xy_values)[:3]))
        self.assertEqual(
            ctx.exception.args[0],
            'Invalid input length. Input should have 8 elements.'
        )

    def test_init__numpy__more_dims(self):
        box = rpbbox.BBox2D(
            np.expand_dims(np.asarray(self.xy_values), axis=0))
        self.assertIsExpectedBox(box)

    def test_init__other(self):
        with self.assertRaises(TypeError) as ctx:
            rpbbox.BBox2D(mock.Mock())
        self.assertEqual(
            ctx.exception.args[0],
            'Expected input to constructor to be a 8 element list, tuple, '
            'numpy ndarray, or BBox2D object.'
        )

    def test_to_list__xyxy(self):
        box = rpbbox.BBox2D(self.xy_values)
        self.assertListEqual(
            box.to_list(),
            list(self.xy_values))

    def test_copy(self):
        box = rpbbox.BBox2D(self.xy_values)
        box2 = box.copy()
        self.assertEqual(box, box2)
        self.assertIsNot(box, box2)

    def test_repr(self):
        box = rpbbox.BBox2D(self.xy_values)
        self.assertEqual(
            str(box),
            'BBox2D({})'.format(box.to_list())
        )

    def test_eq__equal(self):
        box1 = rpbbox.BBox2D(self.xy_values)
        box2 = rpbbox.BBox2D(self.xy_values)
        self.assertIsExpectedBox(box1)
        self.assertIsExpectedBox(box2)
        self.assertTrue(box1 == box2)
        self.assertIsNot(box1, box2)

    def test_eq__equal__invalid_type(self):
        box = rpbbox.BBox2D(self.xy_values)
        self.assertFalse(box == 1)

    def test_serialize(self):
        box1 = rpbbox.BBox2D(self.xy_values)
        state = box1.__getstate__()
        self.assertIsInstance(state, dict)
        pickled = pickle.dumps(state)  # Check it is actually serializable

        box2 = rpbbox.BBox2D([0, 0, 0, 0, 0, 0, 0, 0])
        box2.__setstate__(pickle.loads(pickled))

        self.assertEqual(box1, box2)

    def test_to_obj(self):
        box = rpbbox.BBox2D(self.xy_values)
        expected_list = [
            dict(x=self.expected_x0, y=self.expected_y0),
            dict(x=self.expected_x1, y=self.expected_y1),
            dict(x=self.expected_x2, y=self.expected_y2),
            dict(x=self.expected_x3, y=self.expected_y3),
        ]
        self.assertListEqual(
            box.to_obj(),
            expected_list
        )
