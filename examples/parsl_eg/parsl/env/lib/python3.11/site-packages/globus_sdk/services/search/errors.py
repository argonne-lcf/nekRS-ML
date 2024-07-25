from __future__ import annotations

import typing as t

import requests

from globus_sdk import exc


class SearchAPIError(exc.GlobusAPIError):
    """
    Error class for the Search API client. In addition to the
    inherited instance variables, provides ``error_data``.

    :ivar error_data: Additional object returned in the error response. May be
                      a dict, list, or None.
    """

    def __init__(self, r: requests.Response) -> None:
        self.error_data: dict[str, t.Any] | None = None
        super().__init__(r)

    def _post_parse_hook(self) -> bool:
        self.error_data = self._dict_data.get("error_data")
        return True
