
class Context:
    """
    context of the project: country, scope, any additional info
    """
    def __init__(self):
        self.data = {}

    def get(self, key: str):
        return self.data.get(key)
