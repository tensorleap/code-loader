from enum import Enum, IntEnum


class LeapDataType(Enum):
    Image = 'Image'
    Text = 'Text'
    Graph = 'Graph'
    Numeric = 'Numeric'


class DatasetInputType(Enum):
    Image = "Image"
    Text = "Text"
    Word = "Word"
    Numeric = "Numeric"
    Time_series = "Time_series"


class DatasetOutputType(Enum):
    Numeric = "Numeric"
    Classes = "Classes"
    Mask = "Mask"


class DatasetMetadataType(Enum):
    float = "float"
    string = "string"
    int = "int"
    boolean = "boolean"


class DataStateType(Enum):
    training = "training"
    validation = "validation"
    test = "test"


class DataStateEnum(IntEnum):
    training = 0
    validation = 1
    test = 2


# todo: handle test not run due to error in pre process and Add to TestingSectionEnum Enum didn't run
class TestingSectionEnum(Enum):
    Warnings = "Warnings"
    Errors = "Errors"
