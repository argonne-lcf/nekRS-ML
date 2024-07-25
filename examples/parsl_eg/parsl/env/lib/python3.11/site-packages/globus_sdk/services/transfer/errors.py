from __future__ import annotations

from globus_sdk import exc


class TransferAPIError(exc.GlobusAPIError):
    """
    Error class for the Transfer API client.
    """
