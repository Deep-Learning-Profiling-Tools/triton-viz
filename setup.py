# setup.py
import subprocess
import tomllib
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


def is_git_repo():
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def is_on_tag():
    """Check if the current commit has a tag."""
    try:
        subprocess.check_output(
            ["git", "describe", "--exact-match", "--tags"], stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def get_git_commit_hash(length=8):
    try:
        cmd = ["git", "rev-parse", f"--short={length}", "HEAD"]
        return "+git{}".format(subprocess.check_output(cmd).strip().decode("utf-8"))
    except Exception:
        return ""


def get_full_git_commit_hash():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(cmd).strip().decode("utf-8")
    except Exception:
        return ""


def get_version_suffix():
    if not is_git_repo():
        return ""  # Not a git repo, no suffix
    if is_on_tag():
        return ""  # On a tag, no suffix
    return get_git_commit_hash()  # Not on a tag, add git hash


def get_base_version():
    """Read base version from pyproject.toml."""
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


BASE_VERSION = get_base_version()
TRITON_VIZ_VERSION = BASE_VERSION + get_version_suffix()


def generate_version_file():
    """Generate triton_viz/version.py file."""
    version_file = Path(__file__).parent / "triton_viz" / "version.py"
    git_version = get_full_git_commit_hash() if is_git_repo() else ""

    content = f'''"""Auto-generated version file. Do not edit."""
__version__ = "{TRITON_VIZ_VERSION}"
git_version = "{git_version}"
'''
    version_file.write_text(content)
    print(f"Generated {version_file} with version {TRITON_VIZ_VERSION}")


class BuildPyCommand(build_py):
    """Custom build command to generate version.py before building."""

    def run(self):
        generate_version_file()
        super().run()


# Generate version file before setup to support editable install
generate_version_file()

setup(cmdclass={"build_py": BuildPyCommand})
