import json

from scripts.prepare_r3_human_audit import select_items


def test_human_audit_packet_is_blinded_and_key_retains_condition(tmp_path) -> None:
    source = tmp_path / "outputs.json"
    source.write_text(
        json.dumps([{"id": "p1", "prompt": "prompt", "outputs": ["a", "b"]}]),
        encoding="utf-8",
    )
    packet, key = select_items([source], ["D1_seed42"], count=2, seed=7)
    assert len(packet) == len(key) == 2
    assert all("condition" not in item for item in packet)
    assert {item["condition"] for item in key} == {"D1_seed42"}
    assert {item["item_id"] for item in packet} == {item["item_id"] for item in key}
