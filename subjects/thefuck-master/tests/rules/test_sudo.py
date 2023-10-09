import pytest
from thefuck.rules.sudo import match, get_new_command
from thefuck.types import Command


@pytest.mark.parametrize('output', [
    'Permission denied',
    'permission denied',
    "npm ERR! Error: EACCES, unlink",
    'requested operation requires superuser privilege',
    'need to be root',
    'need root',
    'shutdown: NOT super-user',
    'Error: This command has to be run with superuser privileges (under the root user on most systems).',
    'updatedb: can not open a temporary file for `/var/lib/mlocate/mlocate.db',
    'must be root',
    'You don\'t have access to the history DB.',
    "error: [Errno 13] Permission denied: '/usr/local/lib/python2.7/dist-packages/ipaddr.py'"])
def test_match(output):
    assert match(Command('', output))


def test_not_match():
    assert not match(Command('', ''))
    assert not match(Command('sudo ls', 'Permission denied'))


@pytest.mark.parametrize('before, after', [
    ('ls', 'sudo ls'),
    ('echo a > b', 'sudo sh -c "echo a > b"'),
    ('echo "a" >> b', 'sudo sh -c "echo \\"a\\" >> b"'),
    ('mkdir && touch a', 'sudo sh -c "mkdir && touch a"')])
def test_get_new_command(before, after):
    assert get_new_command(Command(before, '')) == after
