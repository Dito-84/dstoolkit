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


def test_matplotlib_loads_only_when_something_is_drawn():
    # compare(), profile() and the checks must run where matplotlib isn't installed,
    # such as a size-limited AWS Lambda package; plotting imports it on first use
    import subprocess
    import sys

    code = (
        "import sys, pandas as pd, dstoolkit as dst\n"
        "a = pd.DataFrame({'n': range(40), 'c': ['x', 'y'] * 20})\n"
        "dst.compare(a, a, verbose=False)\n"
        "dst.profile(a, charts=False)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib'))\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
