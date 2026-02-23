from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    def __contains__(cls, x):
        try:
            cls(x)
        except ValueError:
            return False
        return True


class Device(str, Enum, metaclass=MetaEnum):
    """
    Represent the device to load the models on
    """
    CUDA = 'cuda'
    CPU = 'cpu'
