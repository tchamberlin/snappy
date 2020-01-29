#! /usr/bin/env python3

"""CLI utility for examining snapshots for a given file

Example usage:

# Show a table of all snapshots for the given file
$ snappy.py /path/to/nfs/file --verbose

# Show the closest 
$ snappy.py /path/to/nfs/file --verbose --target-date "2020 1 1 03:00"

"""

from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import argparse
import hashlib
import logging
import operator
import sys
import subprocess
from tqdm import tqdm

from dateutil import parser as dp
from tabulate import tabulate


logger = logging.getLogger(__name__)

SEARCH_DIRECTIONS = namedtuple("SEARCH_DIRECTIONS", ("near", "before", "after"))(
    "near", "before", "after"
)


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
    sha1 = hashlib.sha1()
    with open(path, "rb") as file:
        while True:
            data = file.read(65536)
            if data:
                sha1.update(data)
            else:
                break
    return sha1.hexdigest()


def format_timedelta(td: timedelta) -> str:
    if td < timedelta(0):
        return "-" + format_timedelta(-td)
    else:
        # Change this to format positive timedeltas the way you want
        return str(td)


def get_closest(iterable: Iterable, target):
    """Return the item in iterable that is closest to the target"""
    if not iterable or target is None:
        return None

    return min(iterable, key=lambda item: abs(item - target))


def get_snapshots(target_path: Path, snap_dir_name: str) -> Tuple[Iterable, Path]:
    mount_path = find_mount(target_path)
    snapshot_dir = mount_path / snap_dir_name
    if not snapshot_dir.exists():
        raise ValueError(
            f"Given path {str(target_path)!r} has no .snapshot directory! Are you "
            "sure it is on NFS (and snapshots are enabled)?"
        )

    return snapshot_dir.iterdir(), mount_path


def get_closest_snapshot_path(
    path: Path,
    search_direction: str,
    snap_dir_name: str,
    target_date=None,
    use_latest_snapshot=None,
    filter_for_existence=False,
) -> Optional[Path]:

    if target_date is None and use_latest_snapshot is None:
        raise ValueError(
            f"Exactly one of target_date (got {target_date}) or use_latest_snapshot "
            f"(got {use_latest_snapshot}) must be given"
        )
    if search_direction not in SEARCH_DIRECTIONS:
        raise AssertionError(f"Unexpected search_direction value: {search_direction}")

    snapshots = parse_snapshots(path, snap_dir_name)
    if target_date is not None:
        if search_direction in ["after", "before"]:
            op = operator.ge if search_direction == "after" else operator.le

            logger.debug(
                f"Limiting search to only snapshots {search_direction.upper()} target date {target_date}"
            )

            snapshots = {
                snapshot_date: snapshot_dir
                for snapshot_date, snapshot_dir in snapshots.items()
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
            f"Found snapshot taken at {closest} ({format_timedelta(delta)} from target): "
            f"{str(snapshots[closest])!r}"
        )

    elif use_latest_snapshot is not None:
        # Get last date
        closest = sorted(snapshots.keys())[-1]
        if closest is None:
            return None

        logger.debug(
            f"Found snapshot taken at {closest} (most recent snapshot): "
            f"{str(snapshots[closest])!r}"
        )
    else:
        raise AssertionError("TODO")

    return snapshots[closest]


def find_mount(path: Path):
    mount_path = path

    while not mount_path.is_mount():
        mount_path = mount_path.parent
    logger.debug(f"Found mount for {path}: {mount_path}")
    return mount_path


def get_date_from_snapshot(snapshot_path: Path):
    return datetime.strptime(snapshot_path.name.split(".")[1], "%Y-%m-%d_%H%M")


def parse_snapshots(target_path: Path, snap_dir_name: str) -> Dict:
    snapshots, mount_path = get_snapshots(target_path, snap_dir_name)
    parsed = {
        get_date_from_snapshot(snapshot_dir): snapshot_dir
        / str(target_path.absolute())[len(str(mount_path.absolute())) + 1 :]
        for snapshot_dir in snapshots
    }

    # logger.debug("These are the snapshots:\n", pformat(parsed))
    return parsed


def hash_snapshots(snapshots: Dict[datetime, str]) -> List[Tuple[str, Optional[str]]]:
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


def dirs_are_identical(a, b):
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
        if len(line) > 0 and line[0] in [">", "<"]:
            logger.debug(f"Change detected; stopping diff: {line}")
            rsync.terminate()
            return False

    # If there are no differences, then return True: dirs are identical
    return True


def print_snapshots(
    path: Path,
    hashed: List[Tuple[str, Optional[str]]],
    quiet=False,
    only_changes=False,
    diff_dirs=False,
    no_progress=False,
) -> None:
    """Print table of known snapshots"""
    if path.is_file():
        current_hash_on_disk = hash_file(path)
    else:
        current_hash_on_disk = None
    table_data = []
    previous_hash = None
    # Disable tqdm if we have explicitly turned off progress, OR if we are NOT
    # diff'ing directories. Directory diffing is the only operation that takes
    # more than 1 second, so no point in having progress the rest of the time
    for snap_path, snap_hash in tqdm(
        hashed, disable=no_progress or not diff_dirs, unit="snapshot", smoothing=1
    ):
        matches_current_hash_on_disk = (
            snap_hash == current_hash_on_disk if current_hash_on_disk else None
        )
        if path.is_file():
            changed = previous_hash != snap_hash if snap_hash else None
        elif diff_dirs:
            changed = not dirs_are_identical(snap_path, path)
        else:
            changed = None

        exists = snap_path.exists()
        if not only_changes or changed:
            table_data.append(
                (snap_path, exists, snap_hash, matches_current_hash_on_disk, changed)
            )
        previous_hash = snap_hash
    valid_snapshots = len([__ for __, the_hash in hashed if the_hash])

    if quiet:
        table = tabulate(table_data, tablefmt="plain")
    else:
        print(f"Found {str(path)!r} in {valid_snapshots}/{len(hashed)} snapshots")
        table = tabulate(
            table_data, headers=("Path", "Exists", "Hash", "Matches Current", "Changed")
        )
    print(table)


def restore_from_snapshot(from_path: Path, to_path: Path, dry_run=False) -> None:
    logger.info(f"To restore from snapshot, run:")
    if from_path.is_dir():
        print(f"$ rsync -a {from_path} {to_path}")
    elif from_path.is_file():
        print(f"$ cp {args}{from_path} {to_path}")
        # logger.debug(f"Copying {from_path} to {to_path}")
        # shutil.copy(from_path, to_path)
    else:
        raise ValueError("wut")


def main() -> None:
    """CLI Main"""
    args = parse_args()
    if args.verbose:
        init_logging(logging.DEBUG)
    else:
        init_logging(logging.INFO)

    if not args.path.is_file():
        logger.debug(f"Given path {str(args.path)!r} is not a file!")

    if args.target_date or args.latest:
        try:
            closest_snapshot_path = get_closest_snapshot_path(
                path=args.path,
                target_date=args.target_date,
                use_latest_snapshot=args.latest,
                search_direction=args.search_direction,
                filter_for_existence=not args.no_check_exists,
                snap_dir_name=args.snap_dir_name,
            )
        except ValueError as error:
            if args.verbose:
                raise
            logger.error(f"Error: {error}")
            sys.exit(1)

        if closest_snapshot_path and closest_snapshot_path.exists():
            # The final printout
            print(closest_snapshot_path)
        else:
            proxy_name = "near"
            if args.after:
                proxy_name = "after"
            elif args.before:
                proxy_name = "before"

            logger.error(f"Could not find snapshot {proxy_name} " f"{args.target_date}")
            sys.exit(1)

        if args.restore:
            restore_from_snapshot(closest_snapshot_path, args.path, args.dry_run)
    else:
        try:
            snapshots = parse_snapshots(args.path, args.snap_dir_name)
        except ValueError as error:
            if args.verbose:
                raise
            logger.error(f"Error: {error}")
            sys.exit(1)

        hashed = hash_snapshots(snapshots)
        if not args.path.is_file() and not args.diff_dirs:
            logger.info(
                f"NOTE: Change detection is disabled! {str(args.path)!r} is a directory; "
                "give --diff-dirs in order to enable directory change detection"
            )
        print_snapshots(
            args.path,
            hashed,
            args.quiet,
            args.only_changes,
            args.diff_dirs,
            args.no_progress,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Locates and describes snapshots ",
    )
    parser.add_argument("path", type=Path, help="The path to examine snapshots of")

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "-d",
        "--date",
        dest="target_date",
        type=dp.parse,
        help="Date to find snapshots closest to",
    )
    date_group.add_argument(
        "--last", dest="latest", action="store_true", help="Select the latest snapshot"
    )
    parser.add_argument(
        "--no-check-exists", action="store_true", help="Turn off existence checks"
    )
    proximity_group = parser.add_mutually_exclusive_group()
    proximity_group.add_argument(
        "-s",
        "--search-direction",
        choices=SEARCH_DIRECTIONS,
        default="before",
        help="Indicate the direction (in time) to search for snapshots, from "
        "the target date",
    )
    proximity_group.add_argument(
        "--after",
        action="store_const",
        const="after",
        dest="search_direction",
        help="If given, only snapshots after the target date will be considered",
    )
    proximity_group.add_argument(
        "--before",
        action="store_const",
        const="before",
        dest="search_direction",
        help="If given, only snapshots before the target date will be considered",
    )
    proximity_group.add_argument(
        "--near",
        action="store_const",
        const="near",
        dest="search_direction",
        help="If given, only snapshots near the target date will be considered",
    )
    parser.add_argument(
        "-c",
        "--only-changes",
        action="store_true",
        help="Show only snapshots which differ from the previous",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase verbosity. This only affects stderr/" "logging output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Decrease verbosity of stdout (should be suitable for processing via awk, etc.). "
        "This has no effect on logging output/stderr.",
    )
    parser.add_argument(
        "-D", "--dry-run", action="store_true", help="Don't make any changes"
    )
    parser.add_argument(
        "--snap-dir-name", help="The name of snapshot directories", default=".snapshot"
    )
    # TODO: Currently only prints cp command!
    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Copy the file from snapshot directory to location on disk",
    )
    parser.add_argument(
        "--diff-dirs",
        action="store_true",
        help="Enable change detection between snapshots of directories. NOTE: "
        "This can be VERY SLOW.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (progress is shown only if --diff-dirs is given)",
    )
    args = parser.parse_args()
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
