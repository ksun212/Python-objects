# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from copy import deepcopy

from pypinyin import pinyin, Style
from pypinyin.style import register, convert


def test_custom_style_with_decorator():
    style_value = 'test_custom_style_with_decorator'

    @register(style_value)
    def func(pinyin, **kwargs):
        return pinyin + str(len(pinyin))

    hans = '北京'
    origin_pinyin_s = pinyin(hans)
    expected_pinyin_s = deepcopy(origin_pinyin_s)
    for pinyin_s in expected_pinyin_s:
        for index, py in enumerate(pinyin_s):
            pinyin_s[index] = func(py)

    assert pinyin(hans, style=style_value) == expected_pinyin_s


def test_custom_style_with_call():
    style_value = 'test_custom_style_with_call'

    def func(pinyin, **kwargs):
        return str(len(pinyin))

    register(style_value, func=func)

    hans = '北京'
    origin_pinyin_s = pinyin(hans)
    expected_pinyin_s = deepcopy(origin_pinyin_s)
    for pinyin_s in expected_pinyin_s:
        for index, py in enumerate(pinyin_s):
            pinyin_s[index] = func(py)

    assert pinyin(hans, style=style_value) == expected_pinyin_s


def test_custom_style_with_han():
    style_value = 'test_custom_style_with_han'

    @register(style_value)
    def func(pinyin, **kwargs):
        return kwargs.get('han', '') + pinyin

    hans = '北京'
    expected_pinyin_s = [['北běi'], ['京jīng']]

    assert pinyin(hans, style=style_value) == expected_pinyin_s


def test_finals_tone3_no_final():
    assert convert('ń', Style.FINALS_TONE3, True, None) == ''
    assert convert('ń', Style.FINALS_TONE3, False, None) == 'n2'


def test_issue_291():
    assert pinyin('誒', style=Style.TONE, heteronym=True) == \
           [['éi', 'xī', 'yì', 'ê̄', 'ế', 'ê̌', 'ěi',
             'ề', 'èi', 'ēi']]
    assert pinyin('誒', style=Style.BOPOMOFO, heteronym=True) == \
           [['ㄟˊ', 'ㄒㄧ', 'ㄧˋ', 'ㄝ', 'ㄝˊ', 'ㄝˇ', 'ㄟˇ',
             'ㄝˋ', 'ㄟˋ', 'ㄟ']]

    assert pinyin('欸', style=Style.TONE, heteronym=True) == \
           [['āi', 'ǎi', 'ê̄', 'ế', 'ê̌', 'ề', 'xiè', 'éi',
             'ěi', 'èi', 'ēi']]
    assert pinyin('欸', style=Style.BOPOMOFO, heteronym=True) == \
           [['ㄞ', 'ㄞˇ', 'ㄝ', 'ㄝˊ', 'ㄝˇ', 'ㄝˋ', 'ㄒㄧㄝˋ', 'ㄟˊ',
             'ㄟˇ', 'ㄟˋ', 'ㄟ']]


if __name__ == '__main__':
    import pytest
    pytest.cmdline.main()
