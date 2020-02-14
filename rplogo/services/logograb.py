"""LogoGrab implementation for text extractor."""
import logging
from http import HTTPStatus
from http.client import ResponseNotReady

from rplogo import extractor
from rplogo import rpbbox
from rplogo.services import http_utils
from rplogo.services import schemas as validation_schemas

LOGGER = logging.getLogger(__name__)


class LogoGrabLogoExtractor(extractor.AbstractLogoExtractor):
    """Extract text from image using LogoGrab service."""

    ENDPOINT = "https://api.logograb.com/detect"

    def __init__(self, developer_key, **kwargs):
        self.developer_key = developer_key
        self.session = self.create_session(developer_key, **kwargs)
        self.endpoint = kwargs.get('endpoint', self.ENDPOINT)

    @staticmethod
    def create_session(
            developer_key,
            **kwargs,
    ):
        kwargs.setdefault('retries', 10)
        kwargs.setdefault('backoff_factor', 0.5)
        s = http_utils.requests_retry_session(**kwargs)
        s.headers.update({
            "X-DEVELOPER-KEY": developer_key,
            "Content-Type": "application/x-www-form-urlencoded"})
        return s

    def _parse_hash(self, response):
        if not isinstance(response, dict):
            raise TypeError('response must be a dict.')
        LOGGER.debug("Reading hash from %s", response)
        response = validation_schemas.validate_request(response)
        return response[response['dataRootName']]

    def _parse_response(self, response):
        if not response:
            raise ValueError('response is empty')

        if not isinstance(response, dict):
            raise TypeError('response must be a dict.')

        response = validation_schemas.validate_response(response)

        LOGGER.debug("Preparing output of %s", response)
        data_root = response.get('dataRootName')
        res = response.get(data_root)['detections']
        session_id = response.get(data_root)['sessionId']
        results = []
        for r in res:
            try:
                logo_name = r['meta']['brand_name']
            except KeyError:
                logo_name = r['name']

            confidence = r.get('confidenceALE')
            if not confidence:
                confidence = r['validationFlags'][0]

            results.append(
                extractor.ExtractionResult(
                    logo_name=logo_name,
                    confidence=confidence,
                    logo=r.get('iconUrl'),
                    box=rpbbox.BBox2D(r.get('coordinates')),
                    session_id=session_id
                )
            )

        return results

    def send_image(self, image_url):
        """Write wrapper to post endpoint logograb."""
        if not isinstance(image_url, str):
            raise TypeError("Invalid image_url type. Should be str")

        LOGGER.info("Sending image %s to %s", image_url, self.endpoint)
        response = self.session.post(
            url=self.endpoint,
            data=dict(mediaUrl=image_url))
        response.raise_for_status()
        return self._parse_hash(response.json())

    def get_result(self, request_hash):
        response = self.session.get(
            url=f"{self.endpoint}/{request_hash}/response")
        response.raise_for_status()
        if response.status_code == HTTPStatus.ACCEPTED.value:
            raise ResponseNotReady
        return self._parse_response(response.json())

    def extract(self, image_url):
        request_data = self.send_image(image_url)
        try:
            return self.get_result(request_data['requestHash'])
        except KeyError:
            return request_data
