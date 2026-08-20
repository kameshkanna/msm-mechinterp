from __future__ import annotations

import json
import os

from msm_mechinterp.reporting import layer_dict_to_json_safe, write_json


def test_layer_dict_to_json_safe() -> None:
    assert layer_dict_to_json_safe({0: 0.5, 12: -0.3}) == {"0": 0.5, "12": -0.3}


def test_write_json_roundtrip(tmp_path) -> None:
    path = os.path.join(str(tmp_path), "nested", "out.json")
    data = {"a": 1, "b": [1, 2, 3], "c": {"0": 0.5}}

    write_json(path, data)

    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data
