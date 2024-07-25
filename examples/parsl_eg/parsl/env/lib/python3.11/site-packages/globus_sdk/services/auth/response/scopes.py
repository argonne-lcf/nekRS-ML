from globus_sdk.response import IterableResponse


class GetScopesResponse(IterableResponse):
    """
    Response class specific to the Get Scopes API

    Provides iteration on the "scopes" array in the response.
    """

    default_iter_key = "scopes"
