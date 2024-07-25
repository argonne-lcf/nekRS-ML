import logging

from globus_sdk import utils

from .base import StaticGlobusAuthorizer

log = logging.getLogger(__name__)


class BasicAuthorizer(StaticGlobusAuthorizer):
    """
    This Authorizer implements Basic Authentication.
    Given a "username" and "password", they are sent base64 encoded in the
    header.

    :param username: Username component for Basic Auth
    :param password: Password component for Basic Auth
    """

    def __init__(self, username: str, password: str) -> None:
        log.info(
            "Setting up a BasicAuthorizer. It will use an "
            "auth type of Basic and cannot handle 401s."
        )
        log.info(f"BasicAuthorizer.username = {username}")
        self.username = username
        self.password = password

        to_b64 = f"{username}:{password}"
        self.header_val = f"Basic {utils.b64str(to_b64)}"
