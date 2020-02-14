"""Tests for logograb extractor."""
import json
import os
import unittest
from http.client import ResponseNotReady
from unittest import mock

import requests
from requests.exceptions import HTTPError

from schema import SchemaError

import rplogo


def mocked_requests_get(*args, **kwargs):
    """Mock response request."""
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

        @staticmethod
        def raise_for_status():
            pass

    return MockResponse(None, 202)


class LogoGrabBase(
        unittest.TestCase):
    """Extract text from image using Amazon Rekognition service."""

    def setUp(self) -> None:
        self.developer = self.get_env_key(
            'LG_DEVELOPER_KEY',
            'jee207947cm5k4j4np1u28hqbkjbh88hotcs0cpl')
        self.lg = rplogo.LogoGrabLogoExtractor(
            developer_key=self.developer,
            max_retries=30,
            backoff_factor=0.3
        )
        self.image_url = 'https://www.gigantes.com/wp-content/'\
                         'uploads/2020/01/kobe-bryant-mamba.jpg'
        super(LogoGrabBase, self).setUp()

    def get_env_key(self, key, default):
        return os.environ.get(key) or default  # allows define key if empty str


class LogoGrabLogoExtractorTests(LogoGrabBase, unittest.TestCase):
    def test_extract_ok(self):
        res = self.lg.extract(self.image_url)
        self.assertIsInstance(res, list)
        self.assertTrue(
            all(isinstance(x, rplogo.ExtractionResult) for x in res)
        )

        self.assertTrue(all(not x.empty for x in res))

    def test_extract_wrong_devkey(self):
        logo_ex = rplogo.LogoGrabLogoExtractor(
            developer_key='rfswe32dvscx',
        )
        with self.assertRaises(HTTPError):
            logo_ex.extract(self.image_url)

    def test_extract_image_wrong_type(self):
        with self.assertRaises(TypeError):
            self.lg.extract(image_url=123)

    def test__parse_hash_type_error(self):
        with self.assertRaises(TypeError):
            self.lg._parse_hash('WRONGTYPE')

    def test_read_ondemand_ok(self):
        self.test__parse_response_ok()

    def test__parse_response_ok(self):
        fpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/sample_response.json'
        )
        resp = json.load(open(fpath, 'r'))
        r = self.lg._parse_response(resp)

        self.assertIsInstance(r, list)
        self.assertEqual(len(r), 1)
        self.assertFalse(r[0].empty)

        self.assertTrue(r[0].confidence > 0.)

    def test__parse_response_ok_meta(self):
        fpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/sample_response_meta.json'
        )
        resp = json.load(open(fpath, 'r'))
        r = self.lg._parse_response(resp)

        fpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/sample_response.json'
        )
        resp = json.load(open(fpath, 'r'))
        r_meta = self.lg._parse_response(resp)

        self.assertFalse(r[0] == r_meta[0])

    def test_parse_response__empty(self):
        with self.assertRaises(ValueError):
            self.lg._parse_response({})

    def test_parse_response__missing_root(self):
        with self.assertRaises(SchemaError):
            self.lg._parse_response(
                {'fakejson': True}
            )

    def test_parse_response_dataroot_none(self):
        with self.assertRaises(SchemaError):
            self.lg._parse_response(
                {'dataRootName': None}
            )

    def test_parse_response__missing_data(self):
        with self.assertRaises(SchemaError):
            self.lg._parse_response(
                {
                    'dataRootName': 'fake',
                    'fake': []
                }
            )


class SendImageTests(LogoGrabBase, unittest.TestCase):

    def test_wront_dev_key(self):
        lg = rplogo.LogoGrabLogoExtractor(
            developer_key=123,
        )
        expected_msg = (
            "Value for header {X-DEVELOPER-KEY: 123} "
            "must be of type str or bytes, not <class 'int'>"
        )
        with self.assertRaises(requests.exceptions.InvalidHeader) as ctx:
            lg.send_image(self.image_url)
        self.assertEqual(
            ctx.exception.args[0],
            expected_msg
        )

    def test_unauthorised(self):
        logo_ex = rplogo.LogoGrabLogoExtractor(
            developer_key='wrong_key',
        )
        with self.assertRaises(HTTPError):
            logo_ex.get_result(self.image_url)


class GetResultTests(LogoGrabBase, unittest.TestCase):

    def test_wront_dev_key(self):
        lg = rplogo.LogoGrabLogoExtractor(
            developer_key=123,
        )
        expected_msg = (
            "Value for header {X-DEVELOPER-KEY: 123} "
            "must be of type str or bytes, not <class 'int'>"
        )
        with self.assertRaises(requests.exceptions.InvalidHeader) as ctx:
            lg.get_result('some-hash')
        self.assertEqual(
            ctx.exception.args[0],
            expected_msg
        )

    def test__parse_response_wrong_type(self):
        with self.assertRaises(TypeError):
            self.lg._parse_response('WRONGTYPE')

    def test_read_ondemand__wronghash(self):
        with self.assertRaises(HTTPError):
            self.lg.get_result('nohash')

    def test_send_ondemand_ok(self):
        res = self.lg.send_image(
            self.image_url
        )
        self.assertIn('requestHash', res)
        self.assertIn('status', res)
        self.assertEqual(res.get('status'), 202)

    @mock.patch('requests.Session.get', side_effect=mocked_requests_get)
    def test_get_result_notready(self, m_get):
        with self.assertRaises(ResponseNotReady):
            self.lg.get_result(
                'valid_hash'
            )
