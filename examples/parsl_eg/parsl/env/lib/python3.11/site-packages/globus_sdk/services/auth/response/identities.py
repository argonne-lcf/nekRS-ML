from globus_sdk.response import IterableResponse


class GetIdentitiesResponse(IterableResponse):
    """
    Response class specific to the Get Identities API

    Provides iteration on the "identities" array in the response.
    """

    default_iter_key = "identities"


class GetIdentityProvidersResponse(IterableResponse):
    """
    Response class specific to the Get Identity Providers API

    Provides iteration on the "identity_providers" array in the response.
    """

    default_iter_key = "identity_providers"
