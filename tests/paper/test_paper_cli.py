from stockbot.cli.paper_arena import build_parser, run_from_args


def test_paper_cli_records_session_without_broker_execution(tmp_path, capsys):
    ledger = tmp_path / "paper.jsonl"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ledger",
            str(ledger),
            "record",
            "--strategy-id",
            "strategy-a",
            "--net-return",
            "0.001",
            "--timestamp",
            "2026-09-08T18:00:00+00:00",
        ]
    )
    assert run_from_args(args) == 0
    output = capsys.readouterr().out
    assert "paper_recorded=" in output
    assert "broker_execution=disabled" in output
    assert ledger.exists()
