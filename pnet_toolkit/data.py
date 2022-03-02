from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Set,
)
import statistics

CSV_SEPARATOR = '|'


@dataclass
class CSVOutputConfig(object):
    arc_weights: bool = False
    filename: bool = False
    filepath: bool = False


@dataclass 
class PNArc(object):
    _id: str
    source: str
    target: str
    weight: int = 1


@dataclass
class PetriNet(object):
    filepath: str
    name: str
    places: Set[str]
    transitions: Set[str]
    arcs: List[PNArc]

    def into_csv(self, config: CSVOutputConfig) -> str:
        arc_weights = tuple(arc.weight for arc in self.arcs)
        fields = []

        if config.filename:
            fields.append(os.path.basename(self.filepath))

        if config.filepath:
            fields.append(os.path.abspath(self.filepath))

        fields.append(self.name)
        fields.append(str(len(self.places)))
        fields.append(str(len(self.transitions)))
        fields.append(str(min(arc_weights)))
        fields.append(str(max(arc_weights)))
        fields.append(str(statistics.mean(arc_weights)))
        fields.append(str(statistics.stdev(arc_weights)))

        if config.arc_weights:
            fields.append(','.join(map(str, arc_weights)))

        return CSV_SEPARATOR.join(fields) 
