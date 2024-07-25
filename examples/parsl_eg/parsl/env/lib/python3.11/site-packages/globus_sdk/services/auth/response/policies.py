from globus_sdk.response import IterableResponse


class GetPoliciesResponse(IterableResponse):
    """
    Response class specific to the Get Policies API

    Provides iteration on the "policies" array in the response.
    """

    default_iter_key = "policies"
