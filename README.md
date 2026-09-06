# NFL First-Quarter Rushing Model

This repository is the reproducible data pipeline behind the 2026 first-quarter rushing opportunity model.

## Goal

Build a season-long reference dataset that combines:

1. Exact 2025 first-quarter rushing results, game by game.
2. Running-back first-quarter rushing results and workload concentration.
3. Opponent first-quarter rushing yards allowed.
4. 2026 Sharp Football Analysis `Rush Efficiency Def` opponent rankings.
5. A transparent 2026 opportunity matrix for identifying favorable early-game rushing environments.

## Data integrity rules

- Historical game-level rushing data comes from nflverse play-by-play via `nflreadpy`.
- Regular season only.
- First quarter only for Q1 output metrics.
- `rush_attempt == 1`.
- No-play penalties are excluded when the `no_play` field is available.
- Team rushing includes all official rushers, including quarterbacks and receivers.
- The RB layer is kept separate and includes roster positions `RB` and `FB`.
- Team-game rows are built from the full 2025 regular-season game skeleton so a zero-rush Q1 is retained as a real zero rather than disappearing from grouped data.
- The build must reconcile the reconstructed 2025 Q1 team rushing totals against the published FootballDB Q1 aggregates before processed outputs are accepted.
- Missing or unmatched player-position records are surfaced explicitly rather than guessed.

## Outputs

The automated build writes these files to `data/processed/`:

- `team_q1_game_log_2025.csv` — 544 team-game rows.
- `team_q1_summary_2025.csv` — offense distribution and hit-rate summary by team.
- `team_q1_defense_summary_2025.csv` — Q1 rushing allowed distribution and hit-rate summary by defense.
- `rb_q1_game_log_2025.csv` — RB/FB game-level Q1 actuals, including zero Q1 yards in games where the player had a rushing attempt later in the game.
- `rb_q1_summary_2025.csv` — player distribution, hit rates, and workload share.
- `reconciliation_2025.csv` — expected vs reconstructed team Q1 rushing yards.
- `unknown_rusher_positions_2025.csv` — rushers that could not be position-matched, if any.

## Build

```bash
python -m pip install -r requirements.txt
python src/build_q1_rushing_dataset.py
```

A GitHub Actions workflow runs the pipeline and commits validated processed outputs back to the repository.

## Attribution

Historical play-by-play and roster data are sourced from the nflverse ecosystem. The historical aggregate reconciliation target is sourced from FootballDB first-quarter team rushing splits. Sharp Football Analysis rankings are maintained separately in the workbook/model because they are proprietary matchup inputs rather than nflverse data.
