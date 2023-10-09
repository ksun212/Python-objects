"""Test the build process with gettext builder with the test root."""

import gettext
import os
import re
import subprocess
from subprocess import CalledProcessError

import pytest

from sphinx.builders.gettext import Catalog, MsgOrigin

try:
    from contextlib import chdir
except ImportError:
    from sphinx.util.osutil import _chdir as chdir


def test_Catalog_duplicated_message():
    catalog = Catalog()
    catalog.add('hello', MsgOrigin('/path/to/filename', 1))
    catalog.add('hello', MsgOrigin('/path/to/filename', 1))
    catalog.add('hello', MsgOrigin('/path/to/filename', 2))
    catalog.add('hello', MsgOrigin('/path/to/yetanother', 1))
    catalog.add('world', MsgOrigin('/path/to/filename', 1))

    assert len(list(catalog)) == 2

    msg1, msg2 = list(catalog)
    assert msg1.text == 'hello'
    assert msg1.locations == [('/path/to/filename', 1),
                              ('/path/to/filename', 2),
                              ('/path/to/yetanother', 1)]
    assert msg2.text == 'world'
    assert msg2.locations == [('/path/to/filename', 1)]


@pytest.mark.sphinx('gettext', srcdir='root-gettext')
def test_build_gettext(app):
    # Generic build; should fail only when the builder is horribly broken.
    app.builder.build_all()

    # Do messages end up in the correct location?
    # top-level documents end up in a message catalog
    assert (app.outdir / 'extapi.pot').isfile()
    # directory items are grouped into sections
    assert (app.outdir / 'subdir.pot').isfile()

    # regression test for issue #960
    catalog = (app.outdir / 'markup.pot').read_text(encoding='utf8')
    assert 'msgid "something, something else, something more"' in catalog


# @pytest.mark.sphinx('gettext', srcdir='root-gettext')
# def test_msgfmt(app):
#     app.builder.build_all()
#     (app.outdir / 'en' / 'LC_MESSAGES').makedirs()
#     with chdir(app.outdir):
#         try:
#             args = ['msginit', '--no-translator', '-i', 'markup.pot', '--locale', 'en_US']
#             subprocess.run(args, capture_output=True, check=True)
#         except OSError:
#             pytest.skip()  # most likely msginit was not found
#         except CalledProcessError as exc:
#             print(exc.stdout)
#             print(exc.stderr)
#             raise AssertionError('msginit exited with return code %s' % exc.returncode)

#         assert (app.outdir / 'en_US.po').isfile(), 'msginit failed'
#         try:
#             args = ['msgfmt', 'en_US.po',
#                     '-o', os.path.join('en', 'LC_MESSAGES', 'test_root.mo')]
#             subprocess.run(args, capture_output=True, check=True)
#         except OSError:
#             pytest.skip()  # most likely msgfmt was not found
#         except CalledProcessError as exc:
#             print(exc.stdout)
#             print(exc.stderr)
#             raise AssertionError('msgfmt exited with return code %s' % exc.returncode)

#         mo = app.outdir / 'en' / 'LC_MESSAGES' / 'test_root.mo'
#         assert mo.isfile(), 'msgfmt failed'

#     _ = gettext.translation('test_root', app.outdir, languages=['en']).gettext
#     assert _("Testing various markup") == "Testing various markup"


@pytest.mark.sphinx(
    'gettext', testroot='intl', srcdir='gettext',
    confoverrides={'gettext_compact': False})
def test_gettext_index_entries(app):
    # regression test for #976
    app.builder.build(['index_entries'])

    _msgid_getter = re.compile(r'msgid "(.*)"').search

    def msgid_getter(msgid):
        m = _msgid_getter(msgid)
        if m:
            return m.groups()[0]
        return None

    pot = (app.outdir / 'index_entries.pot').read_text(encoding='utf8')
    msgids = [_f for _f in map(msgid_getter, pot.splitlines()) if _f]

    expected_msgids = [
        "i18n with index entries",
        "index target section",
        "this is :index:`Newsletter` target paragraph.",
        "various index entries",
        "That's all.",
        "Mailing List",
        "Newsletter",
        "Recipients List",
        "First",
        "Second",
        "Third",
        "Entry",
        "See",
        "Module",
        "Keyword",
        "Operator",
        "Object",
        "Exception",
        "Statement",
        "Builtin",
    ]
    for expect in expected_msgids:
        assert expect in msgids
        msgids.remove(expect)

    # unexpected msgid existent
    assert msgids == []


@pytest.mark.sphinx(
    'gettext', testroot='intl', srcdir='gettext',
    confoverrides={'gettext_compact': False,
                   'gettext_additional_targets': []})
def test_gettext_disable_index_entries(app):
    # regression test for #976
    app.env._pickled_doctree_cache.clear()  # clear cache
    app.builder.build(['index_entries'])

    _msgid_getter = re.compile(r'msgid "(.*)"').search

    def msgid_getter(msgid):
        m = _msgid_getter(msgid)
        if m:
            return m.groups()[0]
        return None

    pot = (app.outdir / 'index_entries.pot').read_text(encoding='utf8')
    msgids = [_f for _f in map(msgid_getter, pot.splitlines()) if _f]

    expected_msgids = [
        "i18n with index entries",
        "index target section",
        "this is :index:`Newsletter` target paragraph.",
        "various index entries",
        "That's all.",
    ]
    for expect in expected_msgids:
        assert expect in msgids
        msgids.remove(expect)

    # unexpected msgid existent
    assert msgids == []


@pytest.mark.sphinx('gettext', testroot='intl', srcdir='gettext')
def test_gettext_template(app):
    app.build()
    assert (app.outdir / 'sphinx.pot').isfile()

    result = (app.outdir / 'sphinx.pot').read_text(encoding='utf8')
    assert "Welcome" in result
    assert "Sphinx %(version)s" in result


@pytest.mark.sphinx('gettext', testroot='gettext-template')
def test_gettext_template_msgid_order_in_sphinxpot(app):
    app.builder.build_all()
    assert (app.outdir / 'sphinx.pot').isfile()

    result = (app.outdir / 'sphinx.pot').read_text(encoding='utf8')
    assert re.search(
        ('msgid "Template 1".*'
         'msgid "This is Template 1\\.".*'
         'msgid "Template 2".*'
         'msgid "This is Template 2\\.".*'),
        result,
        flags=re.S)


@pytest.mark.sphinx(
    'gettext', srcdir='root-gettext',
    confoverrides={'gettext_compact': 'documentation'})
def test_build_single_pot(app):
    app.builder.build_all()

    assert (app.outdir / 'documentation.pot').isfile()

    result = (app.outdir / 'documentation.pot').read_text(encoding='utf8')
    assert re.search(
        ('msgid "Todo".*'
         'msgid "Like footnotes.".*'
         'msgid "The minute.".*'
         'msgid "Generated section".*'),
        result,
        flags=re.S)
