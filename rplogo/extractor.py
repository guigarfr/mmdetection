"""Logo interfaces (extractor and result)."""
from abc import ABC
from abc import abstractmethod

from rplogo import rpbbox


class AbstractLogoExtractor(ABC):
    """Text extractor interface."""

    @abstractmethod
    def extract(self, image_url):
        pass


class ExtractionResult(object):
    """Defines a result from a text extraction."""

    def __init__(
        self, logo_name='', confidence=0.0,
        logo='', box=None, session_id=''
    ):

        self._logo_name = logo_name
        if 0.0 <= confidence <= 1.0:
            self._confidence = confidence
        else:
            raise ValueError(
                'confidence must be between 0 and 1.'
            )
        self._box = box
        self._logo = logo
        self._id = session_id

        if self._box is not None:
            if not isinstance(self._box, rpbbox.BBox2D):
                raise TypeError("Bounding box should be a `BBox2D`")

    def __eq__(self, other):
        if isinstance(other, ExtractionResult):
            return all([
                self.logo_name == other.logo_name,
                self.confidence == other.confidence,
                self.box == other.box,
                self.logo == other.logo,
                self.session_id == other.session_id
            ])
        return False

    @property
    def empty(self):
        return all([
            self.logo_name == '',
            self.confidence == 0.0,
            self.logo == '',
            self.box is None,
            self.session_id == ''
        ])

    def __getstate__(self):
        result = dict(
            logo_name=self._logo_name,
            confidence=self.confidence,
            logo=self.logo,
            session_id=self.session_id
        )
        if self.box:
            result.update({'box': self.box.__getstate__()})

        return result

    def __setstate__(self, state):

        box_state = state.get('box')
        if box_state is None:
            box = None
        else:
            box = rpbbox.BBox2D([0, 0, 0, 0, 0, 0, 0, 0])
            box.__setstate__(box_state)

        self.__init__(
            logo_name=state.get('logo_name', None),
            confidence=state.get('confidence', 0.0),
            logo=state.get('logo', None),
            box=box,
            session_id=state.get('session_id', None)
        )

    def __str__(self):
        text_str = '\"{}\"'.format(self.logo_name)
        conf_str = '{0:.2f}%'.format(self.confidence if self.confidence else 0)
        logo_str = f"{self.logo}"
        if self.box:
            box_str = str(self.box)
        else:
            box_str = '(no box)'
        return f"[{' - '.join([text_str, conf_str, box_str, logo_str])}]"

    def to_dict(self):
        obj = dict(
            logo_name=self._logo_name,
            confidence=self.confidence,
            logo=self.logo,
            session_id=self.session_id
        )
        if self.box:
            obj.update({'box': self.box.to_obj()})

        return obj

    @property
    def logo_name(self):
        return self._logo_name

    @property
    def confidence(self):
        return self._confidence

    @property
    def box(self):
        return self._box

    @property
    def logo(self):
        return self._logo

    @property
    def session_id(self):
        return self._id
