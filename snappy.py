#! /usr/bin/env python3

"""CLI utility for examining snapshots for a given file

Example usage:

# Show a table of all snapshots for the given file
$ snappy.py /path/to/nfs/file --verbose

# Show the closest 
$ snappy.py /path/to/nfs/file --verbose --target-date "2020 1 1 03:00"

"""

from typing import NamedTuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional
import argparse
import csv
import hashlib
import io
import logging
import operator
import re
import shutil
import subprocess
import sys

from dateutil import parser as dp
from tabulate import tabulate
from tqdm import tqdm


logger = logging.getLogger(__name__)


class SearchDirections(NamedTuple):
    after: str = "after"
    before: str = "before"
    near: str = "near"


SEARCH_DIRECTIONS = SearchDirections()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            # self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def hash_file(path) -> str:
    """Compute sha1 hash of file at given path"""
    sha1 = hashlib.sha1()
    with open(path, "rb") as file:
        while True:
            data = file.read(65536)
            if data:
                sha1.update(data)
            else:
                break
    return sha1.hexdigest()


def abs_timedelta(td: timedelta) -> str:
    """Format negative timedeltas sensibly"""
    if td < timedelta(0):
        return "-" + abs_timedelta(-td)
    else:
        # Change this to format positive timedeltas the way you want
        return str(td)


def get_closest(iterable: Iterable, target):
    """Return the item in iterable that is closest to the target"""
    if not iterable or target is None:
        return None

    return min(iterable, key=lambda item: abs(item - target))


def get_snapshots(
    target_path: Path, snap_dir_name: str, snap_filter=None
) -> Tuple[List[Any], int, Any]:
    """Derive mount path and all snapshots from given path"""
    mount_path = find_mount(target_path)
    snapshot_dir = mount_path / snap_dir_name
    if not snapshot_dir.exists():
        raise ValueError(
            f"Given path {str(target_path)!r} has no {snap_dir_name} directory! Are you "
            "sure it is on NFS (and snapshots are enabled)?"
        )

    snapshots = list(snapshot_dir.iterdir())
    total_snapshots = len(snapshots)

    if snap_filter:
        snapshots = [
            path for path in snapshots if str(snapshot_dir / snap_filter) in str(path)
        ]
    return snapshots, total_snapshots, mount_path


def get_closest_snapshot_path(
    snapshots,
    search_direction: str,
    target_date: Optional[datetime],
    filter_for_existence=True,
) -> Optional[Path]:
    """Return the closest snapshot to the target, or None if there aren't any
    
    path: The target path
    search_direction: Search either before the target, after, or both
    target_date: If given, this is used as the tareget, and search_direction is
                 understood to mean temporal proximity
    filter_for_existence: Filter out snapshots that don't contain the given path
    """
    if target_date is None:
        return snapshots[sorted(snapshots)[-1]]

    if search_direction not in SEARCH_DIRECTIONS:
        raise ValueError(f"Unexpected search_direction value: {search_direction}")

    if search_direction in [SEARCH_DIRECTIONS.after, SEARCH_DIRECTIONS.before]:
        op = operator.ge if search_direction == SEARCH_DIRECTIONS.after else operator.le

        logger.debug(
            f"Limiting search to only snapshots {search_direction.upper()} target date {target_date}"
        )

        snapshots = {
            snapshot_date: full_path
            for snapshot_date, full_path in snapshots.items()
            if op(snapshot_date, target_date)
        }

    if filter_for_existence:
        snapshots = {
            snapshot_date: full_path
            for snapshot_date, full_path in snapshots.items()
            if full_path.exists()
        }
    closest = get_closest(snapshots, target_date)
    if closest is None:
        return None
    delta = closest - target_date
    logger.debug(
        f"Found snapshot taken at {closest} ({abs_timedelta(delta)} from target): "
        f"{str(snapshots[closest])!r}"
    )

    return snapshots[closest]


def find_mount(path: Path):
    """Given a path, derive its mountpoint path"""
    mount_path = path

    while not mount_path.is_mount():
        mount_path = mount_path.parent
    logger.debug(f"Found mount for {path}: {mount_path}")

    return mount_path


def get_date_from_snapshot_regex(snapshot_path: Path, snapshot_regex: re.Pattern):
    """Derive date from snapshot directory name using given regex"""
    match = snapshot_regex.match(str(snapshot_path))
    if not match:
        raise ValueError(
            f"Failed to parse date from {str(snapshot_path)!r} with "
            f"given regex {snapshot_regex}"
        )
    return datetime(**{key: int(value) for key, value in match.groupdict().items()})


def rebase_path(old_base, new_base, target_path):
    old_base_length = len(str(old_base.absolute())) + 1
    # Cut off the mount point from the beginning of the target path...
    path_without_old_base = str(target_path)[old_base_length:]
    # ...and replace it with the snapshot directory
    path_with_new_base = new_base / path_without_old_base
    logger.debug(f"Rebased {str(target_path)!r} to {str(path_with_new_base)!r}")
    return path_with_new_base


def parse_snapshots(
    target_path: Path,
    snap_dir_name: str,
    snap_date_regex: re.Pattern,
    rebase_links=True,
    snap_filter=None,
) -> Tuple[Dict[Any, Any], int]:
    snapshots, total_snapshots, mount_path = get_snapshots(
        target_path, snap_dir_name, snap_filter=snap_filter
    )
    parsed = {}

    rebase_summary = None
    for snapshot_dir in snapshots:
        date = get_date_from_snapshot_regex(snapshot_dir, snap_date_regex)
        snapshot_path = rebase_path(mount_path, snapshot_dir, target_path)
        if snap_dir_name not in str(snapshot_path.resolve()):
            if rebase_links:
                new_snapshot_path = rebase_path(
                    mount_path, snapshot_dir, snapshot_path.resolve()
                )
                snapshot_path = new_snapshot_path
                if not rebase_summary:
                    rebase_summary = Path(
                        mount_path, snap_dir_name, "**", snapshot_path.name
                    )
            else:
                raise ValueError(
                    f"Snapshot directory {str(snapshot_path)!r} resolves to a "
                    f"non-snapshot directory: {str(snapshot_path.resolve())!r}. "
                    "Raising error due to presence of --no-rebase-snapshot-links"
                )
        parsed[date] = snapshot_path

    if rebase_summary:
        logger.info(
            f"Snapshots for {str(target_path)!r} contain symlinks to the live filesystem, "
            f"rebasing {str(target_path)!r} -> {str(rebase_summary)!r}"
        )

    return parsed, total_snapshots


def hash_snapshots(snapshots: Dict[datetime, Path]) -> List[Tuple[Path, Optional[str]]]:
    """Derive SHA1 hashes of all snapshots"""
    hashed = []
    file_hash: Optional[str] = None
    for __, full_path in sorted(snapshots.items()):
        if full_path.is_file():
            try:
                file_hash = hash_file(full_path)
            except FileNotFoundError:
                file_hash = None
        else:
            file_hash = None

        hashed.append((full_path, file_hash))

    return hashed


def dirs_are_identical(a: Path, b: Path) -> bool:
    """Determine whether two directories are identical

    This is based solely on file metadata, not contents. rsync is used
    to perform the diff. Diff stops on first difference.
    """

    logger.debug(f"Diffing {str(a)!r} against {str(b)!r}")
    # Note that we are _not_ doing checksums. This takes forever!
    cmd = [
        "rsync",
        # Don't make any changes!
        "--dry-run",
        # Come up with list of file changes
        "--itemize-changes",
        "--archive",
        # This is needed since pathlib strips / off the end, and we need
        # rsync to follow links correctly
        str(a) + "/",
        # This is needed since pathlib strips / off the end, and we need
        # rsync to follow links correctly
        str(b) + "/",
    ]

    logger.debug(f"Executing command: {' '.join(cmd)}")
    rsync = subprocess.Popen(
        cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Read every line from the rsync command, in real time
    for line in rsync.stdout:
        # If any line starts with a character indicating a change,
        # stop the rsync process (we only care about whether the directories
        # differ, not how they differ)
        if line and line[0] in [">", "<"]:
            logger.debug(f"Stopping diff; change detected: {line!r}")
            rsync.terminate()
            return False

    logger.debug(f"Directories are identical")
    # If there are no differences, then return True: dirs are identical
    return True


def print_snapshots(
    path: Path,
    hashed: List[Tuple[Path, Optional[str]]],
    total_snapshots: int,
    csv_output=False,
    only_changes=False,
    diff_dirs=False,
    no_progress=False,
) -> None:
    """Print table of known snapshots. Also derive changes between them."""

    current_hash_on_disk: Optional[str] = None
    if path.is_file():
        current_hash_on_disk = hash_file(path)
    else:
        # We can only hash files!
        current_hash_on_disk = None
    table_data = []
    previous_hash = None
    # Disable tqdm if we have explicitly turned off progress, OR if we are NOT
    # diff'ing directories. Directory diffing is the only operation that takes
    # more than 1 second, so no point in having progress the rest of the time
    logger.info(f"Calculating changes between {str(path)!r} and snapshots...")
    progress = tqdm(
        hashed, disable=no_progress or not diff_dirs, unit="snapshot", smoothing=1
    )
    previous_snap_path = None
    for snap_path, snap_hash in progress:
        progress.set_description(snap_path.parent.name)
        if path.is_file():
            matches_live_system = (
                snap_hash == current_hash_on_disk if current_hash_on_disk else None
            )
        elif diff_dirs:
            matches_live_system = dirs_are_identical(path, snap_path)
        else:
            matches_live_system = None

        if path.is_file():
            # A file has changed if its hash is different than the previous hash
            # If there is no previous hash, then we can't know if the file has changed,
            # se we indicate none. This should only be the case for the earliest snapshot
            changed = (
                previous_hash and previous_hash != snap_hash if snap_hash else None
            )
        elif diff_dirs and previous_snap_path:
            changed = not dirs_are_identical(previous_snap_path, snap_path)
        else:
            # If not a file AND we haven't turned on directory diff'ing, then we can't
            # know whether there has been a change
            changed = None

        exists = snap_path.exists()
        if not only_changes or changed:
            table_data.append(
                (str(snap_path), exists, snap_hash, matches_live_system, changed)
            )
        previous_hash = snap_hash
        previous_snap_path = snap_path

    valid_snapshots = len([None for row in table_data if row[1]])

    if csv_output:
        file = io.StringIO()
        csvwriter = csv.writer(file)
        csvwriter.writerows(table_data)
        table = file.getvalue()
    else:
        total_str = (
            f" (filtered from {total_snapshots} total)"
            if total_snapshots != len(hashed)
            else ""
        )
        print(
            f"Found {str(path)!r} in {valid_snapshots}/{len(hashed)} snapshots{total_str}"
        )
        table = tabulate(
            table_data, headers=("Path", "Exists", "Hash", "Matches Current", "Changed")
        )
    print(table)


def restore_from_snapshot(from_path: Path, to_path: Path, dry_run=False) -> None:
    """Restore requested path from snapshot"""
    if not dry_run:
        logger.info("Snapshot restores are not yet implemented!")
    print(f"Example restore command:")
    if from_path.resolve().is_dir():
        print(f"  # Make {str(to_path)!r} exactly the same as {str(from_path)!r}")
        print(f"  $ rsync --archive --delete {str(from_path) + '/'!r} {str(to_path)!r}")
    else:
        print(f"  $ cp {str(from_path)!r} {str(to_path)!r}")
        # logger.debug(f"Copying {from_path} to {to_path}")
        # shutil.copy(from_path, to_path)


def main() -> None:
    """CLI Main"""
    args = parse_args()
    if args.verbose:
        init_logging(logging.DEBUG)
    elif args.quiet:
        init_logging(logging.WARNING)
    else:
        init_logging(logging.INFO)

    if not args.path.exists():
        logger.error(f"Given path {str(args.path)!r} does not exist!")
        sys.exit(1)

    if args.path.is_file() and args.diff_dirs:
        logger.error(f"{str(args.path)!r} is a file; --diff-dirs cannot be used!")
        sys.exit(1)

    if args.follow_symlink:
        target_path = args.path.resolve()
        logger.error(f"Resolved given path {str(args.path)!r} to {str(target_path)!r}")
        if not target_path.exists():
            logger.error(f"Resolved path {str(args.path)!r} does not exist!")
            sys.exit(1)
    else:
        target_path = args.path
        if args.path != args.path.resolve():
            logger.info(
                f"{str(args.path)!r} contains a symlink! Resolves to {str(args.path.resolve())!r}. "
                "Continuing as-is, but consider using --follow-symlink to resolve this before "
                "processing"
            )

    try:
        snapshots, total_snapshots = parse_snapshots(
            target_path,
            snap_dir_name=args.snap_dir_name,
            snap_date_regex=args.snap_date_regex,
            rebase_links=not args.no_rebase_snapshot_links,
            snap_filter=args.snapshot_type,
        )
    except ValueError as error:
        if args.verbose:
            raise
        logger.error(f"Error: {error}")
        sys.exit(1)

    if not snapshots:
        logger.error(
            f"No snapshots selected for processing (out of {total_snapshots} total)!"
        )
        sys.exit(1)

    if len(snapshots) != total_snapshots:
        logger.info(
            f"Processing only {len(snapshots)}/{total_snapshots} total snapshots"
        )

    if args.target_date or args.restore:
        if args.restore and not args.target_date:
            target_date = None
        else:
            target_date = args.target_date

        try:
            closest_snapshot_path = get_closest_snapshot_path(
                snapshots=snapshots,
                target_date=target_date,
                search_direction=args.search_direction,
            )
        except ValueError as error:
            if args.verbose:
                raise
            logger.error(f"Error: {error}")
            sys.exit(1)

        if closest_snapshot_path and closest_snapshot_path.exists():
            if args.quiet:
                print(closest_snapshot_path)
            else:
                if args.search_direction in (
                    SEARCH_DIRECTIONS.before,
                    SEARCH_DIRECTIONS.after,
                ):
                    first = (
                        "Newest"
                        if args.search_direction == SEARCH_DIRECTIONS.before
                        else "Oldest"
                    )
                    verb = f"{args.search_direction} {target_date} "
                else:
                    first = "Closest"
                    verb = f"to {target_date} "

                if target_date is None:
                    verb = ""
                logger.info(
                    f"{first} snapshot {verb}" f"is {str(closest_snapshot_path)!r}"
                )
        else:
            logger.error(
                f"Could not find snapshot {args.search_direction} {target_date} "
                f"containing {str(target_path)!r}"
            )
            sys.exit(1)

        if args.restore:
            restore_from_snapshot(closest_snapshot_path, target_path, args.dry_run)
    else:
        hashed = hash_snapshots(snapshots)
        if not target_path.is_file() and not args.diff_dirs:
            logger.info(
                f"NOTE: Change detection is disabled because {str(target_path)!r} is a directory! "
                "Try again with --diff-dirs in order to enable directory change detection"
            )

        print_snapshots(
            path=target_path,
            hashed=hashed,
            total_snapshots=total_snapshots,
            csv_output=args.csv,
            only_changes=args.only_changes,
            diff_dirs=args.diff_dirs,
            no_progress=args.no_progress,
        )


class WideHelpFormatter(argparse.HelpFormatter):
    """Formatter that _actually_ fits the console"""

    def __init__(self, *args, **kwargs):
        # If we can't determine terminal size, just let argparse derive it itself
        # in the super class
        width, __ = shutil.get_terminal_size(fallback=(None, None))
        if width:
            kwargs["width"] = width
        super().__init__(*args, **kwargs)


class SelectDatetime(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, "search_direction", self.dest)
        try:
            target_date = dp.parse(values)
        except dp.ParserError as error:
            parser.error(f"Error parsing --{self.dest}: {error}")
        setattr(namespace, "target_date", target_date)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze snapshot history for a given path. "
        "Optionally restore from a given snapshot",
        formatter_class=WideHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=lambda x: Path(x).expanduser(),
        help="The path to examine snapshots of",
    )

    selection_group = parser.add_argument_group("selection arguments")
    selection_group.add_argument(
        "--only-exists",
        action="store_true",
        help="Filter out snapshots that do not include the target path",
    )
    selection_group.add_argument(
        "--before",
        "--after",
        "--near",
        metavar="DATETIME",
        action=SelectDatetime,
        help="Indicates both the direction to search and the date to search for. --after will "
        "return the newest snapshot occurring after/on the given datetime. --before will return "
        "the oldest snapshot occurring before/on the given datetime. --near will return "
        "the closest snapshot to the given datetime, in either direction. Datetimes can be given "
        "in any reasonable format.",
    )
    selection_group.add_argument(
        "-t",
        "--type",
        dest="snapshot_type",
        metavar="SNAPSHOT_TYPE",
        help="Indicate the type/granularity of snapshots to query. "
        "Examples (will vary depending on snapshot setup): weekly, daily, hourly",
    )
    selection_group.add_argument(
        "--no-rebase-snapshot-links",
        action="store_true",
        help="Typically, snapshot paths that link to non-snapshot paths are rebased "
        "so that the point to the correct path in the snapshot, instead of the live filesystem. "
        "Use this to disable that behavior.",
    )

    action_group = parser.add_argument_group("action arguments")
    action_group.add_argument(
        "-D", "--dry-run", action="store_true", help="Don't make any changes"
    )
    action_group.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Copy the file from snapshot directory to location on disk",
    )

    input_group = parser.add_argument_group("input arguments")
    input_group.add_argument(
        "--snap-dir-name",
        help="The name of snapshot directories on your system (default: %(default)s)",
        default=".snapshot",
    )
    input_group.add_argument(
        "--snap-date-regex",
        # https://regex101.com/r/ZOSNFQ/1
        default=r"^.*\w+\.(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<hour>\d{2})(?P<minute>\d{2})$",
        type=re.compile,
        help="Regular expression for parsing the date from the snapshot name. "
        "If you need to update this, your regex MUST contain sufficient groups "
        "to create a datetime from its constructor (default: '%(default)s')",
    )
    input_group.add_argument(
        "-L",
        "--follow-symlink",
        action="store_true",
        help="Fully resolve the given path before processing. "
        "NOTE: This does NOT resolve any snapshot'd symlinks",
    )
    output_group = parser.add_argument_group("output arguments")
    output_group.add_argument(
        "--diff-dirs",
        action="store_true",
        help="Enable change detection between snapshots of directories. NOTE: "
        "This can be VERY SLOW -- diff stops after first difference, but if there "
        "are no differences then the entire directory trees will be diff'd",
    )
    output_group.add_argument(
        "-c",
        "--only-changes",
        action="store_true",
        help="Show only snapshots which differ from the previous",
    )
    output_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase verbosity. This only affects stderr/logging output.",
    )
    output_group.add_argument(
        "--csv", action="store_true", help="Output snapshot summary in csv"
    )
    output_group.add_argument(
        "-q", "--quiet", action="store_true", help="Decrease verbosity of logging"
    )
    output_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (progress is shown only if --diff-dirs is given)",
    )

    args = parser.parse_args()
    if args.csv and (args.target_date or args.restore):
        parser.error("--csv has no effect if not generating a summary!")

    # Need to put this in there ourselves if it hasn't been put in by SelectDatetime
    if not hasattr(args, "target_date"):
        setattr(args, "target_date", None)
    # Need to put this in there ourselves if it hasn't been put in by SelectDatetime
    if not hasattr(args, "search_direction"):
        setattr(args, "search_direction", None)
    return args


def init_logging(level):
    """Initialize logging"""
    logging.getLogger().setLevel(level)
    _logger = logging.getLogger(__name__)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(console_handler)
    _logger.setLevel(level)


if __name__ == "__main__":
    main()
#
