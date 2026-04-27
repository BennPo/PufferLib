import sys
import types

from pufferlib import pufferl
from pufferlib import torch_pufferl


def _native_backend(**attrs):
    defaults = {
        'env_name': 'drone',
        'create_pufferl': object(),
        'rollouts': object(),
        'log': object(),
        'close': object(),
        'load_weights': object(),
        'render': object(),
    }
    defaults.update(attrs)
    return types.SimpleNamespace(**defaults)


def test_resolve_backend_prefers_native_when_complete(monkeypatch):
    native = _native_backend()
    monkeypatch.setattr(pufferl, '_C', native)

    backend = pufferl._resolve_backend({'env_name': 'drone'}, allow_torch_fallback=True)

    assert backend is native


def test_resolve_backend_falls_back_for_eval_when_native_is_incomplete(monkeypatch, capsys):
    native = _native_backend()
    delattr(native, 'create_pufferl')
    monkeypatch.setattr(pufferl, '_C', native)

    dummy_torch_backend = object()
    torch_backend_module = types.ModuleType('pufferlib.torch_pufferl')
    torch_backend_module.PuffeRL = dummy_torch_backend
    monkeypatch.setitem(sys.modules, 'pufferlib.torch_pufferl', torch_backend_module)

    backend = pufferl._resolve_backend({'env_name': 'drone'}, allow_torch_fallback=True)

    assert backend is dummy_torch_backend
    assert 'missing: create_pufferl' in capsys.readouterr().out


def test_load_policy_checkpoint_prefers_native_for_exact_size(monkeypatch, tmp_path):
    path = tmp_path / 'checkpoint.bin'
    path.write_bytes(b'abcdefgh')
    calls = []

    monkeypatch.setattr(torch_pufferl, '_native_checkpoint_num_floats', lambda policy: 2)
    monkeypatch.setattr(
        torch_pufferl,
        '_load_native_checkpoint',
        lambda policy, path, device: calls.append(('native', path, device)),
    )
    monkeypatch.setattr(
        torch_pufferl.torch,
        'load',
        lambda path, map_location: calls.append(('torch', path, map_location)),
    )

    torch_pufferl._load_policy_checkpoint(object(), path, 'cpu')

    assert calls == [('native', path, 'cpu')]


def test_load_policy_checkpoint_falls_back_to_torch_when_native_size_parse_fails(monkeypatch, tmp_path):
    path = tmp_path / 'checkpoint.bin'
    path.write_bytes(b'abcdefgh')
    calls = []

    class Policy:
        def load_state_dict(self, state_dict):
            calls.append(('load_state_dict', state_dict))

    def load_native(policy, path, device):
        calls.append(('native', path, device))
        raise RuntimeError('not native')

    monkeypatch.setattr(torch_pufferl, '_native_checkpoint_num_floats', lambda policy: 2)
    monkeypatch.setattr(torch_pufferl, '_load_native_checkpoint', load_native)
    monkeypatch.setattr(
        torch_pufferl.torch,
        'load',
        lambda path, map_location: {'module.weight': 'tensor'},
    )

    torch_pufferl._load_policy_checkpoint(Policy(), path, 'cpu')

    assert calls == [
        ('native', path, 'cpu'),
        ('load_state_dict', {'weight': 'tensor'}),
    ]


def test_load_policy_checkpoint_uses_torch_for_non_native_size(monkeypatch, tmp_path):
    path = tmp_path / 'checkpoint.bin'
    path.write_bytes(b'abcd')
    calls = []

    class Policy:
        def load_state_dict(self, state_dict):
            calls.append(('load_state_dict', state_dict))

    monkeypatch.setattr(torch_pufferl, '_native_checkpoint_num_floats', lambda policy: 2)
    monkeypatch.setattr(
        torch_pufferl,
        '_load_native_checkpoint',
        lambda policy, path, device: calls.append(('native', path, device)),
    )
    monkeypatch.setattr(
        torch_pufferl.torch,
        'load',
        lambda path, map_location: {'weight': 'tensor'},
    )

    torch_pufferl._load_policy_checkpoint(Policy(), path, 'cpu')

    assert calls == [('load_state_dict', {'weight': 'tensor'})]


def test_load_policy_checkpoint_handles_torch_index_error(monkeypatch, tmp_path):
    path = tmp_path / 'checkpoint.bin'
    path.write_bytes(b'abcd')
    calls = []

    monkeypatch.setattr(torch_pufferl, '_native_checkpoint_num_floats', lambda policy: 2)
    monkeypatch.setattr(
        torch_pufferl,
        '_load_native_checkpoint',
        lambda policy, path, device: calls.append(('native', path, device)),
    )

    def torch_load(path, map_location):
        raise IndexError('pop from empty list')

    monkeypatch.setattr(torch_pufferl.torch, 'load', torch_load)

    torch_pufferl._load_policy_checkpoint(object(), path, 'cpu')

    assert calls == [('native', path, 'cpu')]
