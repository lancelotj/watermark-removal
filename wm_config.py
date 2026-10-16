"""Minimal YAML configuration loader.

Replaces ``neuralgym.Config`` for this project. ``inpaint.yml`` only uses
plain scalars, lists and nested maps, so a small loader is enough and removes
the neuralgym dependency from the inference path.
"""
import yaml


class DictAsMember(dict):
    """Dictionary whose keys are also reachable as attributes.

    Nested dictionaries are wrapped lazily so ``flags.padding`` and
    ``flags.data_flist`` (a list) both behave as expected.
    """

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def Config(filename):
    """Load ``filename`` into a :class:`DictAsMember`."""
    with open(filename, "r") as handle:
        return DictAsMember(yaml.safe_load(handle))
