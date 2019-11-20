class AlreadyFitError(Exception):
    '''An Exception to be thrown when trying to fit an object that was already fit'''
    def __init__(self, *args, **kwargs):
        if not any([args, kwargs]):
            super().__init__('Object already fit')
        else:
            super().__init__(*args, **kwargs)

class NotFitError(Exception):
    '''An Exception to be thrown when an object is not fit'''
    def __init__(self, *args, **kwargs):
        if not any([args, kwargs]):
            super().__init__('Object not fit')
        else:
            super().__init__(*args, **kwargs)
