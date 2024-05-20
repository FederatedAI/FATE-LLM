import time

import click

from ..utils.config import parse_config, default_eval_config
from ..utils.config import _set_namespace


def parse_custom_type(value):
    parts = value.split('=')
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    elif len(parts) == 2 and isinstance(parts[1], str):
        return parts[0], parts[1]
    else:
        raise click.BadParameter('Invalid input format. Use "str=int" or "str=str".')


class LlmSharedOptions(object):
    _options = {
        "eval_config": (('-c', '--eval_config'),
                        dict(type=click.Path(exists=True), help=f"Manual specify config file", default=None),
                        default_eval_config().__str__()),
        "yes": (('-y', '--yes',), dict(type=bool, is_flag=True, help="Skip double check", default=None),
                False),
        "namespace": (('-n', '--namespace'),
                      dict(type=str, help=f"Manual specify fate llm namespace", default=None),
                      time.strftime('%Y%m%d%H%M%S'))
    }

    def __init__(self):
        self._options_kwargs = {}

    def __getitem__(self, item):
        return self._options_kwargs[item]

    def get(self, k, default=None):
        v = self._options_kwargs.get(k, default)
        if v is None and k in self._options:
            v = self._options[k][2]
        return v

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self._options_kwargs[k] = v

    def post_process(self):
        # add defaults here
        for k, v in self._options.items():
            if self._options_kwargs.get(k, None) is None:
                self._options_kwargs[k] = v[2]

        # update config
        config = parse_config(self._options_kwargs['eval_config'])
        self._options_kwargs['eval_config'] = config

        _set_namespace(self._options_kwargs['namespace'])

    @classmethod
    def get_shared_options(cls, hidden=False):
        def shared_options(f):
            for name, option in cls._options.items():
                f = click.option(*option[0], **dict(option[1], hidden=hidden))(f)
            return f

        return shared_options
