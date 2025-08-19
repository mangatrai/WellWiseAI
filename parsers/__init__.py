# parsers/__init__.py
# This file makes the parsers directory a Python package.

from .dlis import parse_dlis
from .las import parse_las
from .csv_parser import parse_csv_file
from .dat import DatParser
# Future parsers can be imported here, e.g.:
# from .asc import parse_asc