from pathlib import Path
import argparse

from style_demo.db import initialize_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the prefilled SQLite demo database")
    parser.add_argument("--force", action="store_true", help="replace an existing database")
    args = parser.parse_args()
    path = Path(__file__).with_name("db.sqlite3")
    initialize_database(path, force=args.force)
    print(f"Database ready: {path}")


if __name__ == "__main__":
    main()

