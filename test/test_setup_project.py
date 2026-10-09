"""Checks for setup_project.py (T01). Run `pytest` from the repo folder, after `python3 setup_project.py` has set it up,
or run this file on its own:

    python test/test_setup_project.py        (or press Run / Debug on this file)

The full check is running setup_project.py itself on the Mac and on Kaggle; these tests cover the two pieces
agreed with the owner, using a temporary folder so nothing real is touched.
"""

import sys
from pathlib import Path

import pytest

# The script being tested lives one folder up. pytest finds it through pytest.ini; this line lets the file
# also run on its own with plain `python`. The import below has to come after it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup_project import PROJECT_FOLDERS, create_project_folders, is_supported_python


def test_every_project_folder_is_created_and_a_second_run_changes_nothing(tmp_path):
    """The first run makes every project folder; the second finds them all and creates nothing."""
    created_first_time = create_project_folders(tmp_path)
    created_second_time = create_project_folders(tmp_path)

    for folder_name in PROJECT_FOLDERS:
        assert (tmp_path / folder_name).is_dir()
    assert len(created_first_time) == len(PROJECT_FOLDERS)
    assert created_second_time == []


def test_python_3_12_and_newer_are_supported_and_older_versions_are_not():
    """3.12 (the Mac) and 3.13 (Kaggle) are accepted; 3.11 and the Mac's built-in 3.9 are refused."""
    assert is_supported_python((3, 12))
    assert is_supported_python((3, 13))
    assert not is_supported_python((3, 11))
    assert not is_supported_python((3, 9))


if __name__ == "__main__":
    # Running this file directly runs its tests, listing each one by name.
    sys.exit(pytest.main([__file__, "-v"]))
