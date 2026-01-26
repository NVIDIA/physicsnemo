# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import pytest
from utils.dataclass_parser import Help, a, parse_args, parse_dict


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.01
    epochs: int = 10
    optional: Optional[bool] = False


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    model_name: str = "default_model"
    opt: a[str, Help("An example option.")] = "a"


@pytest.mark.parametrize("convert_underscore_to_hyphen", [True, False])
def test_parse_args(convert_underscore_to_hyphen):
    # Usage example
    sep = "-" if convert_underscore_to_hyphen else "_"
    args = [
        f"--model.learning{sep}rate",
        "0.1",
        "--model.epochs",
        "20",
        f"--model{sep}name",
        "my_model",
    ]

    with pytest.raises(SystemExit):
        parse_args(
            Config,
            [f"--model.learning{sep}rate", "not a num"],
            convert_underscore_to_hyphen=convert_underscore_to_hyphen,
        )

    expected = Config(
        model=ModelConfig(learning_rate=0.1, epochs=20), model_name="my_model"
    )
    assert (
        parse_args(
            Config, args, convert_underscore_to_hyphen=convert_underscore_to_hyphen
        )
        == expected
    )


def test_parse_dict():
    obj = {"model_name": "hello", "model": {"learning_rate": 0.1}}
    expected = Config(model=ModelConfig(learning_rate=0.1), model_name="hello")
    assert parse_dict(Config, obj) == expected

    with pytest.raises(ValueError):
        parse_dict(Config, {"model_name": 1})


def test_parse_args_optional():
    @dataclass
    class Config:
        a: Optional[int] = None

    c = parse_args(Config, ["--a", "1"])
    assert c == Config(1)


def test_parse_args_union():
    @dataclass
    class Config:
        a: int | None = None

    c = parse_args(Config, ["--a", "1"])
    assert c == Config(1)


def test_parse_args_any():
    @dataclass
    class Config:
        a: Any = None

    c = parse_args(Config, ["--a", "1"])
    assert c == Config("1")


def test_parse_args_bool_default_false():
    @dataclass
    class Config:
        a: bool = False

    c = parse_args(Config, ["--a"])
    assert c == Config(True)


def test_parse_args_bool_default_true():
    @dataclass
    class Config:
        a: bool = True

    c = parse_args(Config, ["--no-a"])
    assert c == Config(False)


def test_parse_args_bool_default_true_nested():
    @dataclass
    class Sub:
        a: bool = False

    @dataclass
    class Config:
        sub: Sub = field(default_factory=lambda: Sub(a=True))

    c = parse_args(Config, ["--sub.no_a"], convert_underscore_to_hyphen=False)
    assert c == Config(Sub(False))

    c = parse_args(Config, ["--sub.no-a"])
    assert c == Config(Sub(False))


def test_enum():
    class Options(Enum):
        a = auto()
        b = auto()

    @dataclass
    class CLI:
        opt: Options = Options.a

    c = parse_args(CLI, ["--opt", "b"])
    c.opt == Options.b

    c = parse_args(CLI, [])
    c.opt == Options.a


def test_parse_args_double_nested():
    @dataclass(eq=True)
    class SubSub:
        a: int = 1

    @dataclass(eq=True)
    class Sub:
        sub: SubSub = field(default_factory=SubSub)

    @dataclass(eq=True)
    class Config:
        sub: Sub = field(default_factory=Sub)

    c = parse_args(Config, ["--sub.sub.a", "1"], convert_underscore_to_hyphen=False)
    assert c == Config()


def test_parse_args_with_list_generic_type():
    """Ensure list[...] | None fields are compatible with parse_args strict checks."""

    @dataclass
    class Config:
        values: list[int] | None = None

    # This should not raise TypeError from isinstance() on a parameterized generic.
    cfg = parse_args(Config, args=[])
    assert cfg.values is None
