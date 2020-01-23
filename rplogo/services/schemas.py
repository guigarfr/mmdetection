"""Validation schemas for logograb jsons."""
import schema

REQUEST_SCHEMA = schema.Schema({
    'dataRootName': str,
}, ignore_extra_keys=True)

REQUEST_DATA_SCHEMA = schema.Schema({
    'requestHash': str,
    'status': int
}, ignore_extra_keys=True)


def validate_request(request):
    """Validate a logograb request json."""
    data = REQUEST_SCHEMA.validate(request)
    data_root = data['dataRootName']
    data_from_root = REQUEST_DATA_SCHEMA.validate(request[data_root])
    data[data_root] = data_from_root
    return data


RESPONSE_SCHEMA = schema.Schema({
    'dataRootName': str,
}, ignore_extra_keys=True)

RESPONSE_DATA_SCHEMA = schema.Schema({
    'detections': [
        schema.Schema({
            'validationFlags': list,
            'coordinates': schema.Schema(
                [float]
            ),
            'name': str,
            'iconUrl': str,
            'id': int,
            schema.Optional('confidenceALE'): schema.Or(int, float, None),
            schema.Optional('meta'): schema.Schema({
                schema.Optional('brand_name'): str
            }, ignore_extra_keys=True)
        }, ignore_extra_keys=True)
    ],
    'sessionId': str,
    'status': int
}, ignore_extra_keys=True)


def validate_response(response):
    """Validate a logograb response."""
    data = RESPONSE_SCHEMA.validate(response)
    data_root = data['dataRootName']
    data_from_root = RESPONSE_DATA_SCHEMA.validate(response[data_root])
    data[data_root] = data_from_root
    return data
