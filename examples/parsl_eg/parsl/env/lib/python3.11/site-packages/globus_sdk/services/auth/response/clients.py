from globus_sdk.response import IterableResponse


class GetClientsResponse(IterableResponse):
    """
    Response class specific to the Get Clients API

    Provides iteration on the "clients" array in the response.
    """

    default_iter_key = "clients"
