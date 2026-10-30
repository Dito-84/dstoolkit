"""Where the one-line summaries go: print (default), logging, or off."""

import contextlib
import io
import logging

import pandas as pd
import pytest

import dstoolkit as dst


@pytest.fixture(autouse=True)
def restore_default():
    yield
    dst.set_message_output("print")


@pytest.fixture
def df():
    return pd.DataFrame({"id": [1, 2, 3], "x": [1.0, None, 3.0]})


def test_default_prints_like_before(df, capsys):
    assert dst.get_message_output() == "print"
    dst.fill_missing(df)
    assert capsys.readouterr().out == "Filled missing values: x (1)\n"


def test_redirect_stdout_still_captures_messages(df):
    # the browser app collects its log this way
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        dst.fill_missing(df)
    assert "Filled missing values" in log.getvalue()


@contextlib.contextmanager
def app_logging(level=logging.INFO):
    """An application's own logging setup: one handler on the root logger."""
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    root, handler = logging.getLogger(), Keep()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(level)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


def test_logging_sends_messages_to_the_apps_logging_setup(df, capsys):
    dst.set_message_output("logging")
    with app_logging() as records:
        dst.fill_missing(df)
        dst.validate(df, {"x": {"min": 0}})
    assert capsys.readouterr().out == ""             # not printed as well
    ours = [r for r in records if r.name == "dstoolkit"]
    assert [r.levelname for r in ours] == ["INFO", "INFO"]
    assert ours[0].getMessage() == "Filled missing values: x (1)"
    assert ours[1].getMessage().startswith("Validated 3 rows")


def test_print_mode_does_not_also_send_to_the_apps_logging(df, capsys):
    with app_logging() as records:
        dst.fill_missing(df)
    assert "Filled missing values" in capsys.readouterr().out
    assert not [r for r in records if r.name == "dstoolkit"]


def test_logging_respects_the_apps_level(df):
    # with the app's logging at WARNING, INFO summaries are dropped, as for any library
    dst.set_message_output("logging")
    with app_logging(logging.WARNING) as records:
        dst.fill_missing(df)
    assert not [r for r in records if r.name == "dstoolkit"]


def test_off_silences_everything(df, capsys):
    dst.set_message_output("off")
    with app_logging(logging.DEBUG) as records:
        dst.fill_missing(df)
    assert capsys.readouterr().out == ""
    assert not [r for r in records if r.name == "dstoolkit"]


def test_verbose_false_silences_one_call_in_every_mode(df, capsys):
    with app_logging() as records:
        for output in ("print", "logging"):
            dst.set_message_output(output)
            dst.fill_missing(df, verbose=False)
    assert capsys.readouterr().out == ""
    assert not [r for r in records if r.name == "dstoolkit"]


def test_switching_back_to_print_does_not_duplicate_messages(df, capsys):
    for output in ("logging", "print", "off", "print", "print"):
        dst.set_message_output(output)
    dst.fill_missing(df)
    assert capsys.readouterr().out.count("Filled missing values") == 1


def test_unknown_output_raises():
    with pytest.raises(ValueError, match='"print", "logging" or "off"'):
        dst.set_message_output("stdout")


def test_reloading_the_package_does_not_print_twice(df, capsys):
    # notebooks often reload modules (importlib.reload, %autoreload); 0.8.0 development
    # versions then attached a second print handler and printed every message twice
    import importlib

    import dstoolkit._messages as messages
    importlib.reload(messages)
    importlib.reload(dst)
    dst.fill_missing(df)
    assert capsys.readouterr().out.count("Filled missing values") == 1
