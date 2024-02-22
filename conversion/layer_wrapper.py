

class BaseWrapper:
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def upper_bound(self, lower_input, upper_input):
        raise NotImplementedError

    def lower_bound(self, lower_input, upper_input):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
