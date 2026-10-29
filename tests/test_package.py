from importlib import resources

import dstoolkit as dst


def test_version_is_set():
    assert dst.__version__.count(".") == 2


def test_every_public_function_is_available_at_top_level():
    for module in (dst.cleaning, dst.exploration, dst.profiling, dst.validation):
        for name in module.__all__:
            assert getattr(dst, name) is getattr(module, name), name


def test_all_lists_only_real_names():
    for name in dst.__all__:
        assert hasattr(dst, name), name


def test_package_ships_type_information():
    assert resources.files("dstoolkit").joinpath("py.typed").is_file()
