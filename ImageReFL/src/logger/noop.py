# src/logger/noop.py
class NoOpWriter:
    """Null object that accepts any writer calls and does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        def no_op(*_args, **_kwargs):
            pass

        return no_op

    def close(self):  # common explicit call
        pass
