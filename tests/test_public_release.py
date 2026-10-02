import pytest

from scripts.check_public_release import validate_files


def test_publication_rejects_internal_data_even_if_allowlisted(tmp_path):
    for name in ("data/review.txt", "private/report.md", "models/metadata.json", "annotations.csv"):
        with pytest.raises(ValueError):
            validate_files(tmp_path, [name])


def test_publication_rejects_link_and_secret(tmp_path):
    source = tmp_path / "public.py"
    source.write_text('print("synthetic fixture")\n')
    validate_files(tmp_path, ["public.py"])
    link = tmp_path / "link.py"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="linked"):
        validate_files(tmp_path, ["link.py"])
    source.write_text("ghp_" + "A" * 40)
    with pytest.raises(ValueError, match="secret"):
        validate_files(tmp_path, ["public.py"])
