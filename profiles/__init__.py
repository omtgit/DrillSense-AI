from .base import Channel, PhysicsFeature, Profile, SeverityRule
from . import drilling, production

PROFILES = {"production": production.PROFILE, "drilling": drilling.PROFILE}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from None
