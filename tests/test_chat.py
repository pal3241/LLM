import pytest

from minillm.chat import format_chat


def test_chat_template() -> None:
    text = format_chat([{"role": "user", "content": "Halo"}])
    assert text == "<|bos|><|user|>\nHalo\n<|assistant|>\n"


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError):
        format_chat([{"role": "hacker", "content": "x"}])
