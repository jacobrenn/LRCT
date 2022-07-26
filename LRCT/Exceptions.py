class NotFitError(Exception):
    '''An Exception to be thrown when an object is not fit'''

    def __init__(self, *args, **kwargs):
        if not any([args, kwargs]):
            super().__init__('Object not fit')
        else:
            super().__init__(*args, **kwargs)
