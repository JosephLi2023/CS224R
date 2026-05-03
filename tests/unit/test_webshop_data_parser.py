"""Unit test for the WebShop setup.sh gdown-id parser.

Validates against the actual upstream princeton-nlp/WebShop setup.sh, which
uses gdown URLs without `-O` and stores the destination filename in a
trailing `# comment`. The parser tracks bash if/elif branches to attribute
each download to the small or all data split.
"""

from __future__ import annotations

from infra.app_data import _parse_gdown_ids


def test_parse_url_form_with_filename_comment() -> None:
    text = (
        "gdown https://drive.google.com/uc?id=1EgHdxQ_dummyid; "
        "# items_shuffle_1000 - product scraped info"
    )
    out = _parse_gdown_ids(text)
    assert out == {
        "items_shuffle_1000.json": {"gid": "1EgHdxQ_dummyid", "split": "common"}
    }


def test_parse_filename_with_explicit_json_extension() -> None:
    text = "gdown https://drive.google.com/uc?id=ABC123XYZQ # items_human_ins.json"
    out = _parse_gdown_ids(text)
    assert out == {
        "items_human_ins.json": {"gid": "ABC123XYZQ", "split": "common"}
    }


def test_parse_real_setup_sh_excerpt() -> None:
    """Mirrors the actual upstream setup.sh structure (verified live)."""
    text = """\
mkdir -p data;
cd data;
if [ "$data" == "small" ]; then
  gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
  gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
elif [ "$data" == "all" ]; then
  gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
  gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
cd ..
"""
    out = _parse_gdown_ids(text)
    # Each filename → its split.
    assert out["items_shuffle_1000.json"] == {
        "gid": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
        "split": "small",
    }
    assert out["items_ins_v2_1000.json"] == {
        "gid": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
        "split": "small",
    }
    assert out["items_shuffle.json"] == {
        "gid": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
        "split": "all",
    }
    assert out["items_ins_v2.json"] == {
        "gid": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
        "split": "all",
    }
    # Common (downloaded regardless of -d flag).
    assert out["items_human_ins.json"] == {
        "gid": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
        "split": "common",
    }


def test_parse_empty_text() -> None:
    assert _parse_gdown_ids("") == {}


def test_parse_no_gdown_invocations() -> None:
    text = "echo hello\nls -la\necho done\n"
    assert _parse_gdown_ids(text) == {}


def test_parse_appends_json_extension_when_missing() -> None:
    text = "gdown https://drive.google.com/uc?id=ABCDE12345 # items_shuffle"
    out = _parse_gdown_ids(text)
    assert "items_shuffle.json" in out


def test_parse_handles_filename_already_with_json() -> None:
    text = "gdown https://drive.google.com/uc?id=XYZ1234567 # items_shuffle.json"
    out = _parse_gdown_ids(text)
    assert "items_shuffle.json" in out
    assert "items_shuffle.json.json" not in out  # no double-extension
