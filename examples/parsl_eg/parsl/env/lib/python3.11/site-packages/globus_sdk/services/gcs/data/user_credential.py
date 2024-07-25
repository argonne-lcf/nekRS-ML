from __future__ import annotations

import typing as t

from globus_sdk import utils
from globus_sdk._types import UUIDLike


class UserCredentialDocument(utils.PayloadWrapper):
    """
    Convenience class for constructing a UserCredential document
    to use as the `data` parameter to `create_user_credential` and
    `update_user_credential`

    :param DATA_TYPE: Versioned document type.
    :param identity_id: UUID of the Globus identity this credential will
        provide access for
    :param connector_id: UUID of the connector this credential is for
    :param username: Username of the local account this credential will provide
        access to, format is connector specific
    :param display_name: Display name for this credential
    :param storage_gateway_id: UUID of the storage gateway this credential is for
    :param policies: Connector specific policies for this credential
    :param additional_fields: Additional data for inclusion in the document
    """

    def __init__(
        self,
        DATA_TYPE: str = "user_credential#1.0.0",
        identity_id: UUIDLike | None = None,
        connector_id: UUIDLike | None = None,
        username: str | None = None,
        display_name: str | None = None,
        storage_gateway_id: UUIDLike | None = None,
        policies: dict[str, t.Any] | None = None,
        additional_fields: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__()
        self._set_optstrs(
            DATA_TYPE=DATA_TYPE,
            identity_id=identity_id,
            connector_id=connector_id,
            username=username,
            display_name=display_name,
            storage_gateway_id=storage_gateway_id,
        )
        self._set_value("policies", policies)

        if additional_fields is not None:
            self.update(additional_fields)
