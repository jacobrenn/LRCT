class AlreadyFitError(Exception):
    def __init__(self, *args, **kwargs):
        if not any([args, kwargs]):
            super().__init__('Object already fit')
        else:
            super().__init__(*args, **kwargs)

class NotFitError(Exception):
    def __init__(self, *args, **kwargs):
        if not any([args, kwargs]):
            super().__init__('Object not fit')
        else:
            super().__init__(*args, **kwargs)