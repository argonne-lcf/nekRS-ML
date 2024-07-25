from globus_sdk.response import IterableResponse


class GetClientCredentialsResponse(IterableResponse):
    """
    Response class specific to the Get Credentials API

    Provides iteration on the "credentials" array in the response.
    """

    default_iter_key = "credentials"
