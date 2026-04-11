import sys
import types

from pufferlib import pufferl


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
