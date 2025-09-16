from .aistudio_operator import make_outbound_call

class Agent():
    """
    Agent class conforming to the operator's expected interface.
    """
    @classmethod
    async def run(cls, kwargs):
        try:
            prompt = kwargs.get("prompt")
            customer_mobile = kwargs.get("mobile_number")
            file = kwargs.get("file")
            response = await make_outbound_call(customer_mobile, file, prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"Error in WebCall agent: {str(e)}") from e