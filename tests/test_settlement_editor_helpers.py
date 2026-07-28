from app import _accounting_player_labels, _settlement_entry_editor_label
from poker_tracker.persistence.models import HandPlayer, SettlementEntry


def _player(key: str, *, name: str = "HJ", position: str = "HJ") -> HandPlayer:
    return HandPlayer(
        hand_id=1,
        player_key=key,
        player_name=name,
        position=position,
        starting_stack=100,
    )


def test_accounting_labels_do_not_repeat_or_collapse_duplicate_players() -> None:
    players = [_player("abcdefgh-1111"), _player("abcdefgh-2222")]

    labels = _accounting_player_labels(players)

    assert len(labels) == 2
    assert all("HJ · HJ" not in label for label in labels)
    assert {player.player_key for player in labels.values()} == {
        "abcdefgh-1111",
        "abcdefgh-2222",
    }


def test_accounting_label_uses_seat_without_repeating_position() -> None:
    player = _player("player-key-0001").model_copy(update={"seat_index": 2})

    labels = _accounting_player_labels([player])

    assert list(labels) == ["Seat 2 · HJ"]


def test_legacy_settlement_entry_resolves_only_an_unambiguous_name() -> None:
    player = _player("player-key-0001", name="Hero", position="BTN")
    labels = _accounting_player_labels([player])
    label_by_key = {saved.player_key: label for label, saved in labels.items()}
    entry = SettlementEntry(
        hand_id=1,
        entry_type="award",
        player_name="Hero",
        pot_index=0,
    )

    assert _settlement_entry_editor_label(entry, labels, label_by_key) in labels


def test_stable_key_wins_over_a_stale_name_snapshot() -> None:
    player = _player("player-key-0001", name="Hero", position="BTN")
    labels = _accounting_player_labels([player])
    label_by_key = {saved.player_key: label for label, saved in labels.items()}
    entry = SettlementEntry(
        hand_id=1,
        entry_type="award",
        player_key=player.player_key,
        player_name="Old hero name",
        pot_index=0,
    )

    assert _settlement_entry_editor_label(entry, labels, label_by_key) in labels


def test_ambiguous_legacy_name_requires_reassignment() -> None:
    players = [
        _player("player-key-0001", name="Alex", position="BTN"),
        _player("player-key-0002", name="Alex", position="BB"),
    ]
    labels = _accounting_player_labels(players)
    label_by_key = {saved.player_key: label for label, saved in labels.items()}
    entry = SettlementEntry(
        hand_id=1,
        entry_type="award",
        player_name="Alex",
        pot_index=0,
    )

    assert _settlement_entry_editor_label(
        entry, labels, label_by_key
    ).startswith("[Needs reassignment]")


def test_stale_settlement_entry_gets_an_explicit_reassignment_label() -> None:
    player = _player("player-key-0001", name="Hero", position="BTN")
    labels = _accounting_player_labels([player])
    label_by_key = {saved.player_key: label for label, saved in labels.items()}
    entry = SettlementEntry(
        hand_id=1,
        entry_type="award",
        player_key="deleted-player-key",
        player_name="Villain",
        pot_index=0,
    )

    rendered = _settlement_entry_editor_label(entry, labels, label_by_key)

    assert rendered.startswith("[Needs reassignment] Villain")
    assert rendered not in labels


def test_unknown_stable_key_is_never_rebound_by_matching_name() -> None:
    player = _player("player-key-0001", name="Hero", position="BTN")
    labels = _accounting_player_labels([player])
    label_by_key = {saved.player_key: label for label, saved in labels.items()}
    entry = SettlementEntry(
        hand_id=1,
        entry_type="award",
        player_key="deleted-player-key",
        player_name="Hero",
        pot_index=0,
    )

    rendered = _settlement_entry_editor_label(entry, labels, label_by_key)

    assert rendered.startswith("[Needs reassignment]")
    assert rendered not in labels
