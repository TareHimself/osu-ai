from enum import Enum


class EModelType(Enum):
    Unknown = 'Unknown'
    Actions = 'Actions'
    Aim = 'Aim'
    Combined = 'Combined'


class EPlayAreaIndices(Enum):
    Width = 0
    Height = 1
    OffsetX = 2
    OffsetY = 3
