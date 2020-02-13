import pickle
import random
import unittest

from rplogo import extractor
from rplogo import rpbbox


class ExtractionResultTests(unittest.TestCase):

    def generate_random_box(self):
        return tuple((random.uniform(0, 1) for _ in range(8)))

    def test_init__empty(self):
        res = extractor.ExtractionResult()
        self.assertEqual('', res.logo_name)
        self.assertEqual(0.0, res.confidence)
        self.assertEqual(None, res.box)
        self.assertTrue(res.empty)

    def test_init__box(self):
        expected_box = rpbbox.BBox2D(
            self.generate_random_box())
        res = extractor.ExtractionResult(box=expected_box)
        self.assertEqual(0.0, res.confidence)
        self.assertEqual('', res.logo_name)
        self.assertEqual(expected_box, res.box)
        self.assertFalse(res.empty)

    def test_init__box__invalid(self):
        with self.assertRaises(TypeError) as ctx:
            extractor.ExtractionResult(box=(1, 2, 3, 4))
        self.assertEqual(
            "Bounding box should be a `BBox2D`", ctx.exception.args[0])

    def test_equal(self):
        box = self.generate_random_box()
        e1 = extractor.ExtractionResult(
            logo_name="hi Guille",
            box=rpbbox.BBox2D(box)
        )
        e2 = extractor.ExtractionResult(
            logo_name="hi Guille",
            box=rpbbox.BBox2D(box)
        )
        self.assertEqual(e1, e2)

    def test_not_equal__type(self):
        self.assertNotEqual(
            extractor.ExtractionResult(logo_name="equality test"),
            rpbbox.BBox2D((0, 0, 0, 0, 0, 0, 0, 0)))

    def test_str__with_box__without_children(self):
        box = self.generate_random_box()
        res = extractor.ExtractionResult(
            logo_name="hi Guille",
            box=rpbbox.BBox2D(box)
        )
        str_box = ', '.join([str(x) for x in box])
        self.assertEqual(
            f"[\"hi Guille\" - 0.00% - BBox2D([{str_box}]) - ]",
            str(res))

    def test_box__without_children(self):
        box = (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
        res = extractor.ExtractionResult(
            logo_name="hi Guille",
            box=rpbbox.BBox2D(box)
        )
        self.assertEqual(res.box.x0, 0.0)
        self.assertEqual(res.box.y1, 1.0)
        self.assertEqual(res.box.x2, 2.0)
        self.assertEqual(res.box.y3, 3.0)

    def test_str__without_box(self):
        res = extractor.ExtractionResult(
            logo_name="hi Guille")
        self.assertEqual(
            "[\"hi Guille\" - 0.00% - (no box) - ]",
            str(res))

    def test_serialize(self):
        res = extractor.ExtractionResult(
            logo_name="hi Guille",
            box=rpbbox.BBox2D(
                self.generate_random_box())
        )
        state = res.__getstate__()
        self.assertIsInstance(state, dict)
        pickled = pickle.dumps(state)  # Check it is actually serializable

        res2 = extractor.ExtractionResult()
        res2.__setstate__(pickle.loads(pickled))

        self.assertEqual(res, res2)

    def test_serialize__no_box(self):
        res = extractor.ExtractionResult(
            logo_name="hi",
        )
        state = res.__getstate__()
        self.assertIsInstance(state, dict)
        pickled = pickle.dumps(state)  # Check it is actually serializable

        res2 = extractor.ExtractionResult()
        res2.__setstate__(pickle.loads(pickled))

        self.assertEqual(res, res2)

    def test_to_dict(self):
        expected_box = rpbbox.BBox2D(
            self.generate_random_box()
        )
        res = extractor.ExtractionResult(
            logo_name='logo',
            box=expected_box,
            confidence=0.5,
            logo='logo.url',
            session_id='session_id'
        )
        expected_dict = dict(
            logo_name='logo',
            box=expected_box.to_obj(),
            confidence=0.5,
            logo='logo.url',
            session_id='session_id'
        )

        self.assertDictEqual(
            res.to_dict(),
            expected_dict
        )

    def test_incorrect_confidence(self):
        with self.assertRaises(ValueError):
            extractor.ExtractionResult(confidence=-1.1)

        with self.assertRaises(ValueError):
            extractor.ExtractionResult(confidence=1.1)
