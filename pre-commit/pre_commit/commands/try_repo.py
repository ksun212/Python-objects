from __future__ import annotations

import argparse
import logging
import os.path
import tempfile

import pre_commit.constants as C
from pre_commit import git
from pre_commit import output
from pre_commit.clientlib import load_manifest
from pre_commit.commands.run import run
from pre_commit.store import Store
from pre_commit.util import cmd_output_b
from pre_commit.xargs import xargs
from pre_commit.yaml import yaml_dump

logger = logging.getLogger(__name__)


def _repo_ref(tmpdir: str, repo: str, ref: str | None) -> tuple[str, str]:
    # if `ref` is explicitly passed, use it
    if ref is not None:
        return repo, ref

    ref = git.head_rev(repo)
    # if it exists on disk, we'll try and clone it with the local changes
    if os.path.exists(repo) and git.has_diff('HEAD', repo=repo):
        logger.warning('Creating temporary repo with uncommitted changes...')

        shadow = os.path.join(tmpdir, 'shadow-repo')
        cmd_output_b('git', 'clone', repo, shadow)
        cmd_output_b('git', 'checkout', ref, '-b', '_pc_tmp', cwd=shadow)

        idx = git.git_path('index', repo=shadow)
        objs = git.git_path('objects', repo=shadow)
        env = dict(os.environ, GIT_INDEX_FILE=idx, GIT_OBJECT_DIRECTORY=objs)

        staged_files = git.get_staged_files(cwd=repo)
        if staged_files:
            xargs(('git', 'add', '--'), staged_files, cwd=repo, env=env)

        cmd_output_b('git', 'add', '-u', cwd=repo, env=env)
        git.commit(repo=shadow)

        return shadow, git.head_rev(shadow)
    else:
        return repo, ref


def try_repo(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tempdir:
        repo, ref = _repo_ref(tempdir, args.repo, args.ref)

        store = Store(tempdir)
        if args.hook:
            hooks = [{'id': args.hook}]
        else:
            repo_path = store.clone(repo, ref)
            manifest = load_manifest(os.path.join(repo_path, C.MANIFEST_FILE))
            manifest = sorted(manifest, key=lambda hook: hook['id'])
            hooks = [{'id': hook['id']} for hook in manifest]

        config = {'repos': [{'repo': repo, 'rev': ref, 'hooks': hooks}]}
        config_s = yaml_dump(config)

        config_filename = os.path.join(tempdir, C.CONFIG_FILE)
        with open(config_filename, 'w') as cfg:
            cfg.write(config_s)

        output.write_line('=' * 79)
        output.write_line('Using config:')
        output.write_line('=' * 79)
        output.write(config_s)
        output.write_line('=' * 79)

        return run(config_filename, store, args)
