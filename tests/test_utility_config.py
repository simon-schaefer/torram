import filecmp
import os
import pathlib
import torram


def test_config_empty():
    config_empty = torram.utility.Config.empty()
    assert len(config_empty.dict) == 0


def test_config_get():
    config = torram.utility.Config({"a": 2})
    assert config.get("a", default=5) == 2  # uses pre-defined value, when given
    assert config.get("b", default=5) == 5  # uses default value when not given


def test_config_get_nested():
    config = torram.utility.Config({"a": {"aa": 2}})
    assert config.get("a/aa") == 2


def test_config_flattening():
    config = torram.utility.Config({"a": {"aa": 2}, "b": 3})
    assert config.dict == {"a/aa": 2, "b": 3}


def test_config_set():
    config = torram.utility.Config.empty()
    config.set("a", 2)
    assert "a" in config.dict
    assert config.dict["a"] == 2


def test_config_from_yaml():
    test_file = pathlib.Path(os.path.realpath(__file__)).parent / "assets" / "config_test.yaml"
    config = torram.utility.Config.from_yaml(test_file)
    assert config.dict == {"a/aa": 2, "b": 3}


def test_config_to_yaml():
    test_file_dir = pathlib.Path(os.path.realpath(__file__)).parent / "assets"
    test_file_read = test_file_dir / "config_test.yaml"
    test_file_write = test_file_dir / "cache" / "config_test.yaml"

    config = torram.utility.Config.from_yaml(test_file_read)
    config.save_yaml(test_file_write)
    assert filecmp.cmp(test_file_read, test_file_write)


def test_config_to_str():
    config = torram.utility.Config({"a": {"aa": 2}})
    assert "aa" in str(config)  # hard to test
