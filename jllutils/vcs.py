from abc import abstractmethod, ABC
import os
import re
import shlex
import subprocess as sp


class VCSError(Exception):
    """General VCS error type
    """
    pass


class VCSExecError(VCSError):
    """Error type for problems running the VCS executable
    """
    pass


class VCSRepoError(VCSError):
    """Error type for problems with a VCS repo
    """


class _NullError(VCSError):
    """Error type to use as placeholder for errors that should not be caught. Never instantiate.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError('Do not instantiate this error type, it must represent a no-op')


class VCS(object):
    """Parent class for interacting with version control systems

    This class contains abstract methods and so generally should not be directly instantiated. It is intended to be
    subclassed with the abstract methods implemented to return the necessary information for a particular version
    control system. See :class:`Git` and :class:`Hg` in this module for examples.
    """
    def __init__(self, repo_dir, cmd=None, ignore_shell_errors=True):
        """Instantiate a VCS instance.

        This instance will be able to report on the state of a particular repository. There are several built-in methods
        to return common information as well as a general `call` method to run arbitrary commands.

        Parameters
        ----------
        repo_dir : pathlike
            A directory in the repository to query. Does not need to be the top level directory, any directory that you
            can call the VCS command (e.g. git or hg) in on the command line will work.

        cmd : str
            The VCS command to use. This may one expected to be found on your PATH (e.g. "git", "hg") or a full path to
            the executable desired (e.g. "/usr/bin/git").

        ignore_shell_errors : bool
            If `True`, then any shell errors that occur when trying to call the VCS are ignored and placeholder values
            (usually "Unknown" or `False`) are returned instead. If `False`, then any error that occurs when calling
            the VCS shell commands will result in a `VCSExecError`.

        Raises
        ------
        VCSExecError
            if the executable given as `cmd` cannot be found and `ignore_shell_errors` is `False`.

        VCSRepoError
            if the given path to the repo is not a directory.
        """
        if not os.path.isdir(repo_dir):
            raise VCSRepoError('Bad repo_dir (not a directory): {}'.format(repo_dir))
        self._cmd = cmd
        self._dir = repo_dir
        self._errs_to_catch = _NullError if not ignore_shell_errors else VCSError
        try:
            sp.call([cmd], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        except FileNotFoundError:
            if not ignore_shell_errors:
                raise VCSExecError('The executable {} does not appear to be on your path'.format(cmd))

    def __call__(self, *args, **kwargs):
        """Alias to :meth:`VCS.call`
        """
        return self.call(*args, **kwargs)

    def call(self, *args, enforce_unicode=True, prepend_command=True):
        """Call an arbitrary command for this VCS in this repo

        Parameters
        ----------
        args
            The command to execute, either as a single string or as individual arguments. Note that if a single string
            is passed, it will be split into arguments using :func:`shlex.split` before being passed to a subprocess
            call. This method will never use `shell=True` in the subprocess call.

            By default, if the first argument does not match the root VCS command given when this instance was
            initialized, that command will be prepended. (For example, if this was a Git VCS, `'log'` becomes
            `'git log'`, but `'git log'` does not become `'git git log'`.) However, you can override this by setting
            `prepend_command` to `False`, in which case the main command is never prepended.

        enforce_unicode : bool
            If `True` (default) the returned stdout of the command will always be returned as a unicode string, not
            bytes. Set this to `False` to return the native type for the subprocess call (bytes or string).

        prepend_command : bool
            If `False`, then the default VCS command for this instance will not be prepended to this command. You will
            have to pass that as part of the full command (e.g. `'git log'` instead of just `'log'`). You can use this
            to override the VCS executable used for a single command.

        Returns
        -------
        str or bytes
            The stdout from the command. If `enforce_unicode` is `True`, it should always be returned as a unicode
            encoded string. Otherwise, the return type will be the native return type for `subprocess` calls.
        """
        if len(args) == 1:
            # Allow for the case of the arguments passed as a single string
            args = tuple(shlex.split(args[0]))
        if args[0] != self._cmd and prepend_command:
            args = (self._cmd,) + args

        try:
            result = sp.check_output(args, cwd=self._dir)
        except Exception as err:
            raise VCSExecError('Error while trying to execute VCS command "{cmd}" in {dir}: {etype}: {emsg}'
                               .format(cmd=' '.join(args), dir=self._dir, etype=type(err).__name__, emsg=str(err)))
        if isinstance(result, bytes) and enforce_unicode:
            result = result.decode('utf8')
        return result.strip()

    @abstractmethod
    def commit_info(self):
        """Get the hash, branch, and date of the parent commit of the current working directory

        Returns
        -------
        hash : str
            The hash of the parent commit.

        branch : str
            The currently checked out branch

        date : str
            The date the parent commit was made. Format not guaranteed.
        """
        pass

    @abstractmethod
    def repo_root(self):
        """Get the top directory of the repository

        Returns
        -------
        root : str
            Path to the top directory of this repo. May be absolute or relative, depending on implementation. If
            an absolute path is required, you must take steps to guarantee that yourself.
        """
        pass

    @abstractmethod
    def is_repo_clean(self):
        """Check if there are uncommitted changes to any tracked files in this repo.

        Returns
        -------
        clean : bool
            `True` if no changes to tracked files, `False` otherwise.
        """
        pass


class Git(VCS):
    def __init__(self, repo, cmd='git'):
        super(Git, self).__init__(repo, cmd=cmd)

    def repo_root(self):
        try:
            return self.call('rev-parse', '--show-toplevel')
        except self._errs_to_catch:
            return 'Unknown'

    def commit_info(self):
        try:
            parent = self.call('rev-parse', '--short', 'HEAD')
            branch = self.call('rev-parse', '--abbrev-ref', 'HEAD')
            date = self.call('show', '-s', '--format=%ci', 'HEAD')
        except self._errs_to_catch:
            return 'Unknown', 'Unknown', 'Unknown'
        else:
            return parent, branch, date

    def is_repo_clean(self):
        try:
            return 0 == sp.call([self._cmd, 'diff-index', '--quiet', 'HEAD', '--'], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        except self._errs_to_catch:
            return False


class Hg(VCS):
    def __init__(self, repo, cmd='hg'):
        super(Hg, self).__init__(repo, cmd=cmd)

    def commit_info(self):
        # Get the last commit (-l 1) in the current branch (-f)
        try:
            summary = self.call('log', '-f', '-l', '1').splitlines()
        except self._errs_to_catch:
            return 'Unknown', 'Unknown', 'Unknown'

        log_dict = dict()

        for line in summary:
            splitline = line.split(':', 1)
            if len(splitline) < 2:
                continue
            k, v = splitline
            log_dict[k.strip()] = v.strip()

        parent = re.search('(?<=:)\\w+', log_dict['changeset']).group()
        # In Mercurial, if on the default branch, then log does not include a branch name in the output
        branch = log_dict['branch'] if 'branch' in log_dict else 'default'
        parent_date = log_dict['date']
        # Convert to unicode strings to avoid them getting formatted as "b'abc'" or "b'default'" in unicode strings
        return parent, branch, parent_date

    def repo_root(self):
        try:
            return self.call('root')
        except self._errs_to_catch:
            return 'Unknown'

    def is_repo_clean(self):
        # -q means it will not print untracked files, so as long as there are no lines, then the repo is clean
        try:
            summary = self.call('status', '-q').splitlines()
        except self._errs_to_catch:
            return False
        else:
            return len(summary) == 0


def init_vcs(cmd, repo_dir):
    vcs_classes = {'git': Git, 'hg': Hg}
    if cmd in vcs_classes:
        vcs_cls = vcs_classes[cmd]
        return vcs_cls(repo=repo_dir)
    else:
        raise ValueError('No VCS subclass defined for {}'.format(cmd))
