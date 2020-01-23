"""Utils module with http functionalities."""
import requests
from requests import adapters
from requests.packages.urllib3.util import retry


def requests_retry_session(**kwargs):
    """Create a retry loop using requests session."""
    session = kwargs.get('session', requests.Session())
    retries = kwargs.get('retries', 3)
    retry_obj = retry.Retry(
        total=kwargs.get('total', retries),
        read=kwargs.get('read', retries),
        connect=kwargs.get('connect', retries),
        backoff_factor=kwargs.get('backoff_factor', 0.3),
        status_forcelist=kwargs.get('status_forcelist', (202, 500, 502, 504)),
    )
    adapter = adapters.HTTPAdapter(max_retries=retry_obj)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
