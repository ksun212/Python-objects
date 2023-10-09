from typing import Optional
class Repo:
    DVC_DIR = ".dvc"

    from dvc.repo.add import add  # type: ignore[misc]
    from dvc.repo.checkout import checkout  # type: ignore[misc]
    from dvc.repo.commit import commit  # type: ignore[misc]
    from dvc.repo.destroy import destroy  # type: ignore[misc]
    from dvc.repo.diff import diff  # type: ignore[misc]
    from dvc.repo.fetch import fetch  # type: ignore[misc]
    from dvc.repo.freeze import freeze, unfreeze  # type: ignore[misc]
    from dvc.repo.gc import gc  # type: ignore[misc]
    from dvc.repo.get import get as _get  # type: ignore[misc]
    from dvc.repo.get_url import get_url as _get_url  # type: ignore[misc]
    from dvc.repo.imp import imp  # type: ignore[misc]
    from dvc.repo.imp_url import imp_url  # type: ignore[misc]
    from dvc.repo.install import install  # type: ignore[misc]
    from dvc.repo.ls import ls as _ls  # type: ignore[misc]
    from dvc.repo.ls_url import ls_url as _ls_url  # type: ignore[misc]
    from dvc.repo.move import move  # type: ignore[misc]
    from dvc.repo.pull import pull  # type: ignore[misc]
    from dvc.repo.push import push  # type: ignore[misc]
    from dvc.repo.remove import remove  # type: ignore[misc]
    from dvc.repo.reproduce import reproduce  # type: ignore[misc]
    from dvc.repo.run import run  # type: ignore[misc]
    from dvc.repo.status import status  # type: ignore[misc]
    from dvc.repo.update import update  # type: ignore[misc]

    from .data import status as data_status  # type: ignore[misc]

    ls = staticmethod(_ls)
    ls_url = staticmethod(_ls_url)
    get = staticmethod(_get)
    get_url = staticmethod(_get_url)

    def _get_repo_dirs(
        self,
        root_dir: Optional[str] = None,
        fs: Optional["FileSystem"] = None,
        uninitialized: bool = False,
        scm: Optional[Union["Git", "NoSCM"]] = None,
    ) -> Tuple[str, Optional[str]]:
        from dvc.fs import localfs
        from dvc.scm import SCM, SCMError

        dvc_dir: Optional[str] = None
        try:
            root_dir = self.find_root(root_dir, fs)
            fs = fs or localfs
            dvc_dir = fs.path.join(root_dir, self.DVC_DIR)
        except NotDvcRepoError:
            if not uninitialized:
                raise

            if not scm:
                try:
                    scm = SCM(root_dir or os.curdir)
                    if scm.dulwich.repo.bare:
                        raise NotDvcRepoError(f"{scm.root_dir} is a bare git repo")
                except SCMError:
                    scm = SCM(os.curdir, no_scm=True)

            if not fs or not root_dir:
                root_dir = scm.root_dir

        assert root_dir
        return root_dir, dvc_dir

    def __init__(  # noqa: PLR0915
        self,
        root_dir: Optional[str] = None,
        fs: Optional["FileSystem"] = None,
        rev: Optional[str] = None,
        subrepos: bool = False,
        uninitialized: bool = False,
        config: Optional["DictStrAny"] = None,
        url: Optional[str] = None,
        repo_factory: Optional[Callable] = None,
        scm: Optional[Union["Git", "NoSCM"]] = None,
    ):
        from dvc.cachemgr import CacheManager
        from dvc.config import Config
        from dvc.data_cloud import DataCloud
        from dvc.fs import GitFileSystem, LocalFileSystem, localfs
        from dvc.lock import LockNoop, make_lock
        from dvc.repo.metrics import Metrics
        from dvc.repo.params import Params
        from dvc.repo.plots import Plots
        from dvc.repo.stage import StageLoad
        from dvc.scm import SCM
        from dvc.stage.cache import StageCache
        from dvc_data.hashfile.state import State, StateNoop

        self.url = url
        self._fs_conf = {"repo_factory": repo_factory}
        self._fs = fs or localfs
        self._scm = scm
        self._data_index = None

        if rev and not fs:
            self._scm = scm = SCM(root_dir or os.curdir)
            root_dir = "/"
            self._fs = GitFileSystem(scm=self._scm, rev=rev)

        self.root_dir: str
        self.dvc_dir: Optional[str]
        (
            self.root_dir,
            self.dvc_dir,
        ) = self._get_repo_dirs(
            root_dir=root_dir,
            fs=self.fs,
            uninitialized=uninitialized,
            scm=scm,
        )

        self.config: Config = Config(self.dvc_dir, fs=self.fs, config=config)
        self._uninitialized = uninitialized

        # used by DVCFileSystem to determine if it should traverse subrepos
        self.subrepos = subrepos

        self.cloud: "DataCloud" = DataCloud(self)
        self.stage: "StageLoad" = StageLoad(self)

        self.lock: "LockBase"
        self.cache: CacheManager
        self.state: "StateBase"
        if isinstance(self.fs, GitFileSystem) or not self.dvc_dir:
            self.lock = LockNoop()
            self.state = StateNoop()
            self.cache = CacheManager(self)
        else:
            if isinstance(self.fs, LocalFileSystem):
                assert self.tmp_dir
                self.fs.makedirs(self.tmp_dir, exist_ok=True)

                self.lock = make_lock(
                    self.fs.path.join(self.tmp_dir, "lock"),
                    tmp_dir=self.tmp_dir,
                    hardlink_lock=self.config["core"].get("hardlink_lock", False),
                    friendly=True,
                )
                os.makedirs(self.site_cache_dir, exist_ok=True)
                self.state = State(self.root_dir, self.site_cache_dir, self.dvcignore)
            else:
                self.lock = LockNoop()
                self.state = StateNoop()

            self.cache = CacheManager(self)

            self.stage_cache = StageCache(self)

            self._ignore()

        self.metrics: Metrics = Metrics(self)
        self.plots: Plots = Plots(self)
        self.params: Params = Params(self)

        self.stage_collection_error_handler: Optional[
            Callable[[str, Exception], None]
        ] = None
        self._lock_depth: int = 0

    def __str__(self):
        return self.url or self.root_dir



class Stage(params.StageParams):
    # pylint:disable=no-value-for-parameter
    # rwlocked() confuses pylint
    repo: Optional[Repo]
    def __init__(  # noqa: PLR0913
        self,
        repo,
        path=None,
        cmd=None,
        wdir=os.curdir,
        deps=None,
        outs=None,
        md5=None,
        locked=False,  # backward compatibility
        frozen=False,
        always_changed=False,
        stage_text=None,
        dvcfile=None,
        desc: Optional[str] = None,
        meta=None,
    ):
        if deps is None:
            deps = []
        if outs is None:
            outs = []

        self.repo = repo
        self._path = path
        self.cmd = cmd
        self.wdir = wdir
        self.outs = outs
        self.deps = deps
        self.md5 = md5
        self.frozen = locked or frozen
        self.always_changed = always_changed
        self._stage_text = stage_text
        self._dvcfile = dvcfile
        self.desc: Optional[str] = desc
        self.meta = meta
        self.raw_data = RawData()


@decorator
def rwlocked(call, read=None, write=None):
    import sys

    from dvc.dependency.repo import RepoDependency
    from dvc.rwlock import rwlock

    if read is None:
        read = []

    if write is None:
        write = []

    stage: Stage = call._args[0]  # pylint: disable=protected-access

    assert stage.repo.lock.is_locked

    def _chain(names):
        return [
            item.fs_path
            for attr in names
            for item in getattr(stage, attr)
            # There is no need to lock RepoDependency deps, as there is no
            # corresponding OutputREPO, so we can't even write it.
            if not isinstance(item, RepoDependency)
        ]

    cmd = " ".join(sys.argv)

    with rwlock(
        stage.repo.tmp_dir,
        stage.repo.fs,
        cmd,
        _chain(read),
        _chain(write),
        stage.repo.config["core"].get("hardlink_lock", False),
    ):
        return call()

