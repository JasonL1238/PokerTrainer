from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import pytest

from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import (
    EXPORT_VERSION,
    export_session,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import reconcile_persisted_hand


def _make_db(path: str | Path = ":memory:") -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _create_hand(db: PokerDatabase, *, name: str = "Accounting") -> tuple[Session, Hand]:
    session = db.create_session(Session(name=name, stakes="1/2 NL"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="No-limit Hold'em",
            blinds_antes="1/2 NL",
        )
    )
    return session, hand


def test_reconciliation_uses_board_runout_for_no_flop_no_drop_rake() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Preflop all-in", stakes="1/2 NL"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="No-limit Hold'em",
            blinds_antes="1/2 NL",
            board_cards="2c 3d 4h 5s 6c",
        )
    )
    for key in ("A", "B"):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key,
                starting_stack=10,
            )
        )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="A",
            player_name="A",
            street="preflop",
            action_type="all-in",
            amount=10,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="B",
            player_name="B",
            street="preflop",
            action_type="call",
            amount=10,
        )
    )
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            status="reconciled",
            rake_rate=0.05,
            no_flop_no_drop=True,
            gross_pot=20,
            rake_amount=1,
            net_pot=19,
            is_balanced=True,
        )
    )
    db.create_settlement_entry(
        SettlementEntry(
            hand_id=hand.id,
            entry_type="award",
            pot_index=0,
            player_key="A",
            player_name="A",
            amount=19,
        )
    )

    reconciliation = reconcile_persisted_hand(db, hand.id)

    assert reconciliation.ledger.rake == pytest.approx(1)
    assert reconciliation.ledger.payouts == pytest.approx({"A": 19, "B": 0})
    assert reconciliation.is_authoritative is True
    db.close()


def _create_v6_database(path: Path) -> None:
    """Create the minimum real legacy schema needed to exercise v7 migration."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO schema_metadata (key, value) VALUES ('schema_version', '6');

        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date_played TEXT NOT NULL,
            platform TEXT NOT NULL DEFAULT '',
            stakes TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE hands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            hand_number INTEGER NOT NULL,
            game_type TEXT NOT NULL DEFAULT '',
            blinds_antes TEXT NOT NULL DEFAULT '',
            table_size INTEGER,
            effective_stack REAL,
            hero_position TEXT NOT NULL DEFAULT '',
            hero_cards TEXT NOT NULL DEFAULT '',
            board_cards TEXT NOT NULL DEFAULT '',
            pot_size REAL,
            result TEXT NOT NULL DEFAULT '',
            hero_bb_won REAL,
            review_status TEXT NOT NULL DEFAULT 'unreviewed',
            confidence_score REAL,
            source_type TEXT NOT NULL DEFAULT 'manual',
            tags TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE hand_players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT NOT NULL DEFAULT '',
            starting_stack REAL,
            is_hero INTEGER NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            street TEXT NOT NULL,
            action_index INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT NOT NULL DEFAULT '',
            action_type TEXT NOT NULL,
            amount REAL,
            pot_before REAL,
            stack_before REAL,
            notes TEXT NOT NULL DEFAULT ''
        );

        INSERT INTO sessions (
            id, name, date_played, platform, stakes, notes, created_at
        ) VALUES (
            1, 'Legacy session', '2026-01-01', 'Manual', '1/2 NL', '',
            '2026-01-01T00:00:00+00:00'
        );
        INSERT INTO hands (
            id, session_id, hand_number, game_type, blinds_antes, table_size,
            effective_stack, hero_position, hero_cards, board_cards, pot_size,
            result, hero_bb_won, review_status, confidence_score, source_type,
            tags, notes, created_at
        ) VALUES (
            1, 1, 1, 'No-limit Hold''em', '1/2 NL', 6, 100, 'BTN',
            'Ah Qs', '', NULL, '', NULL, 'unreviewed', NULL, 'manual', '[]', '',
            '2026-01-01T00:00:00+00:00'
        );
        INSERT INTO hand_players (
            id, hand_id, player_name, position, starting_stack, is_hero, notes
        ) VALUES (7, 1, 'Hero', 'BTN', 100, 1, '');
        INSERT INTO actions (
            id, hand_id, street, action_index, player_name, position,
            action_type, amount, pot_before, stack_before, notes
        ) VALUES (11, 1, 'preflop', 1, 'Hero', 'BTN', 'raise', 8, NULL, NULL, '');
        """
    )
    connection.commit()
    connection.close()


def test_v6_rows_migrate_without_claiming_ambiguous_amount_semantics(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-v6.sqlite3"
    _create_v6_database(database_path)

    db = _make_db(database_path)
    player = db.fetch_players_by_hand(1)[0]
    action = db.fetch_actions_by_hand(1)[0]

    assert db.schema_version() == SCHEMA_VERSION
    assert SCHEMA_VERSION >= 7
    assert player.player_key
    assert player.seat_index is None
    assert action.amount == pytest.approx(8)
    assert action.amount_semantics == "unknown"
    # A legacy display name remains available, but it is not promoted into an
    # authoritative identity link when the old row did not contain one.
    assert action.player_name == "Hero"
    db.close()

    reopened = _make_db(database_path)
    assert reopened.fetch_players_by_hand(1)[0].player_key == player.player_key
    assert reopened.fetch_actions_by_hand(1)[0].amount_semantics == "unknown"
    reopened.close()


def test_new_actions_are_explicitly_incremental_and_keep_stable_seat_identity() -> None:
    db = _make_db()
    _, hand = _create_hand(db)
    players = [
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="seat-3",
                seat_index=3,
                player_name="Alex",
                position="CO",
                starting_stack=100,
            )
        ),
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="seat-7",
                seat_index=7,
                player_name="Alex",
                position="BB",
                starting_stack=100,
            )
        ),
    ]
    actions = [
        db.create_action(
            Action(
                hand_id=hand.id,
                street="preflop",
                player_key="seat-3",
                player_name="CO display label",
                position="CO",
                action_type="ante",
                amount=0.25,
            )
        ),
        db.create_action(
            Action(
                hand_id=hand.id,
                street="preflop",
                player_key="seat-7",
                player_name="BB display label",
                position="BB",
                action_type="post_blind",
                amount=1,
                amount_semantics="incremental",
            )
        ),
        db.create_action(
            Action(
                hand_id=hand.id,
                street="preflop",
                player_key="seat-3",
                player_name="CO display label",
                position="CO",
                action_type="raise",
                amount=3,
                amount_semantics="incremental",
            )
        ),
    ]

    assert [player.player_key for player in players] == ["seat-3", "seat-7"]
    assert [player.seat_index for player in players] == [3, 7]
    assert actions[0].amount_semantics == "incremental"
    assert [action.player_key for action in actions] == ["seat-3", "seat-7", "seat-3"]

    fetched_players = {player.player_key: player for player in db.fetch_players_by_hand(hand.id)}
    fetched_actions = db.fetch_actions_by_hand(hand.id)
    assert set(fetched_players) == {"seat-3", "seat-7"}
    assert fetched_players["seat-3"].seat_index == 3
    assert fetched_players["seat-7"].seat_index == 7
    assert [action.action_type for action in fetched_actions] == [
        "ante",
        "post_blind",
        "raise",
    ]
    assert all(action.amount_semantics == "incremental" for action in fetched_actions)
    # Identity is the player key, not a mutable or duplicated display name.
    assert fetched_actions[0].player_key == fetched_actions[2].player_key
    assert fetched_players[fetched_actions[0].player_key].position == "CO"
    db.close()


def _persist_complex_settlement(
    db: PokerDatabase,
    hand: Hand,
) -> tuple[HandSettlement, list[SettlementEntry]]:
    settlement = HandSettlement(
        hand_id=hand.id,
        status="settled",
        dead_money=0,
        rake_rate=0.05,
        rake_cap=5,
        rake_rounding_unit=0.5,
        no_flop_no_drop=False,
        gross_pot=280,
        rake_amount=5,
        net_pot=275,
        is_balanced=True,
        warnings=[],
    )
    saved = db.upsert_hand_settlement(settlement)
    entries = [
        SettlementEntry(
            hand_id=hand.id,
            entry_type="refund",
            pot_index=None,
            player_key="A",
            player_name="A",
            amount=20,
            entry_order=1,
        ),
        # Main pot is split, with its two awards kept separately.
        SettlementEntry(
            hand_id=hand.id,
            entry_type="award",
            pot_index=0,
            player_key="C",
            player_name="C",
            amount=37.5,
            entry_order=2,
        ),
        SettlementEntry(
            hand_id=hand.id,
            entry_type="award",
            pot_index=0,
            player_key="D",
            player_name="D",
            amount=37.5,
            entry_order=3,
        ),
        SettlementEntry(
            hand_id=hand.id,
            entry_type="award",
            pot_index=1,
            player_key="B",
            player_name="B",
            amount=120,
            entry_order=4,
        ),
        SettlementEntry(
            hand_id=hand.id,
            entry_type="award",
            pot_index=2,
            player_key="A",
            player_name="A",
            amount=80,
            entry_order=5,
        ),
    ]
    db.replace_settlement_entries(hand.id, entries)
    return saved, db.fetch_settlement_entries(hand.id)


def test_settlement_round_trip_preserves_refunds_rake_and_split_side_pot_awards() -> None:
    db = _make_db()
    _, hand = _create_hand(db)
    contributions = {"A": 120, "B": 60, "C": 20, "D": 100}
    for seat, (player_key, amount) in enumerate(contributions.items()):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=player_key,
                seat_index=seat,
                player_name=player_key,
                starting_stack=amount,
            )
        )
        db.create_action(
            Action(
                hand_id=hand.id,
                street="turn",
                player_key=player_key,
                player_name=player_key,
                action_type="all-in",
                amount=amount,
                amount_semantics="incremental",
            )
        )

    saved, entries = _persist_complex_settlement(db, hand)
    fetched = db.fetch_hand_settlement(hand.id)

    assert fetched == saved
    assert fetched.status == "settled"
    assert fetched.rake_rate == pytest.approx(0.05)
    assert fetched.rake_cap == pytest.approx(5)
    assert fetched.rake_rounding_unit == pytest.approx(0.5)
    assert fetched.gross_pot == pytest.approx(280)
    assert fetched.rake_amount == pytest.approx(5)
    assert fetched.net_pot == pytest.approx(275)
    assert fetched.is_balanced is True
    refunds = [entry for entry in entries if entry.entry_type == "refund"]
    awards = [entry for entry in entries if entry.entry_type == "award"]
    assert len(refunds) == 1
    assert refunds[0].pot_index is None
    assert refunds[0].player_key == "A"
    assert refunds[0].amount == pytest.approx(20)
    assert [entry.pot_index for entry in awards] == [0, 0, 1, 2]
    assert [entry.player_key for entry in awards[:2]] == ["C", "D"]
    assert [entry.amount for entry in awards] == pytest.approx([37.5, 37.5, 120, 80])
    # Fetching may group awards before refunds, but the declared order remains
    # stable for odd-chip/split-pot precedence and export round trips.
    assert [
        (entry.entry_type, entry.pot_index, entry.player_key, entry.entry_order)
        for entry in entries
    ] == [
        ("award", 0, "C", 2),
        ("award", 0, "D", 3),
        ("award", 1, "B", 4),
        ("award", 2, "A", 5),
        ("refund", None, "A", 1),
    ]

    _assert_reconciled(db, hand.id)
    db.close()


def _assert_reconciled(db: PokerDatabase, hand_id: int) -> None:
    settlement = db.fetch_hand_settlement(hand_id)
    actions = db.fetch_actions_by_hand(hand_id)
    entries = db.fetch_settlement_entries(hand_id)

    contributions: defaultdict[str, float] = defaultdict(float)
    refunds: defaultdict[str, float] = defaultdict(float)
    awards: defaultdict[str, float] = defaultdict(float)
    for action in actions:
        assert action.amount_semantics == "incremental"
        contributions[action.player_key] += action.amount or 0
    for entry in entries:
        destination = refunds if entry.entry_type == "refund" else awards
        destination[entry.player_key] += entry.amount

    assert sum(contributions.values()) - sum(refunds.values()) == pytest.approx(
        settlement.gross_pot
    )
    assert settlement.gross_pot - settlement.rake_amount == pytest.approx(
        settlement.net_pot
    )
    assert sum(awards.values()) == pytest.approx(settlement.net_pot)

    player_keys = set(contributions) | set(refunds) | set(awards)
    net_results = {
        key: awards[key] + refunds[key] - contributions[key] for key in player_keys
    }
    assert sum(net_results.values()) + settlement.rake_amount == pytest.approx(0)
    assert settlement.is_balanced is True


def test_settlement_and_identity_survive_export_import_round_trip() -> None:
    source = _make_db()
    session, hand = _create_hand(source, name="Ledger export")
    for seat, player_key in enumerate(("A", "B")):
        source.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=player_key,
                seat_index=seat,
                player_name=f"Player {player_key}",
                starting_stack=50,
            )
        )
        source.create_action(
            Action(
                hand_id=hand.id,
                street="river",
                player_key=player_key,
                player_name=f"Player {player_key}",
                action_type="all-in",
                amount=50,
                amount_semantics="incremental",
            )
        )
    source.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            status="settled",
            gross_pot=100,
            rake_amount=2,
            net_pot=98,
            rake_rate=0.02,
            rake_cap=3,
            is_balanced=True,
        )
    )
    source.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="B",
                player_name="Player B",
                amount=98,
                entry_order=1,
            )
        ],
    )

    payload = export_session(source, session.id)
    assert EXPORT_VERSION >= 2
    assert payload["hands"][0]["settlement"]["rake_amount"] == pytest.approx(2)
    assert payload["hands"][0]["settlement_entries"][0]["player_key"] == "B"

    target = _make_db()
    imported_session = import_session(target, payload)
    imported_hand = target.fetch_hands_by_session(imported_session.id)[0]
    imported_players = target.fetch_players_by_hand(imported_hand.id)
    imported_actions = target.fetch_actions_by_hand(imported_hand.id)
    imported_settlement = target.fetch_hand_settlement(imported_hand.id)
    imported_entries = target.fetch_settlement_entries(imported_hand.id)

    assert [(player.player_key, player.seat_index) for player in imported_players] == [
        ("A", 0),
        ("B", 1),
    ]
    assert [action.player_key for action in imported_actions] == ["A", "B"]
    assert all(action.amount_semantics == "incremental" for action in imported_actions)
    assert imported_settlement.gross_pot == pytest.approx(100)
    assert imported_settlement.rake_amount == pytest.approx(2)
    assert imported_settlement.net_pot == pytest.approx(98)
    assert len(imported_entries) == 1
    assert imported_entries[0].player_key == "B"
    assert imported_entries[0].amount == pytest.approx(98)
    _assert_reconciled(target, imported_hand.id)

    source.close()
    target.close()


def test_legacy_v1_import_stays_inspectable_but_not_authoritative() -> None:
    legacy_payload = {
        "export_version": 1,
        "session": Session(name="Legacy JSON").model_dump(mode="json"),
        "hands": [
            {
                "hand": Hand(session_id=999, hand_number=1).model_dump(mode="json"),
                "players": [
                    {
                        "hand_id": 999,
                        "player_name": "Hero",
                        "position": "BTN",
                        "starting_stack": 100,
                        "is_hero": True,
                        "notes": "",
                    }
                ],
                "actions": [
                    {
                        "hand_id": 999,
                        "street": "preflop",
                        "action_index": 1,
                        "player_name": "Hero",
                        "position": "BTN",
                        "action_type": "raise",
                        "amount": 8,
                        "pot_before": None,
                        "stack_before": None,
                        "notes": "",
                    }
                ],
                "reviews": [],
            }
        ],
    }

    db = _make_db()
    imported = import_session(db, legacy_payload)
    hand = db.fetch_hands_by_session(imported.id)[0]
    player = db.fetch_players_by_hand(hand.id)[0]
    action = db.fetch_actions_by_hand(hand.id)[0]

    assert player.player_key
    assert player.seat_index is None
    assert action.amount == pytest.approx(8)
    assert action.amount_semantics == "unknown"
    assert db.fetch_hand_settlement(hand.id) is None
    assert db.fetch_settlement_entries(hand.id) == []
    db.close()
