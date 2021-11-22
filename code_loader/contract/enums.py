from enum import Enum


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
