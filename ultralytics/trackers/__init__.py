# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
<<<<<<< HEAD
from .track import register_tracker

__all__ = "BOTSORT", "BYTETracker", "register_tracker"  # allow simpler import
=======
from .deep_oc_sort import DeepOCSORT
from .fast_tracker import FASTTracker
from .oc_sort import OCSORT
from .track import register_tracker
from .track_tracker import TRACKTRACK

__all__ = (
    "BOTSORT",
    "OCSORT",
    "TRACKTRACK",
    "BYTETracker",
    "DeepOCSORT",
    "FASTTracker",
    "register_tracker",
)  # allow simpler import
>>>>>>> origin/main
