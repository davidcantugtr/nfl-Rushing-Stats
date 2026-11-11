from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def cell(value):
    if pd.isna(value):
        return {}
    if isinstance(value, bool):
        return {"userEnteredValue": {"boolValue": bool(value)}}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"userEnteredValue": {"numberValue": float(value)}}
    return {"userEnteredValue": {"stringValue": str(value)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("sheet_id", type=int)
    ap.add_argument("header_row", type=int, help="1-based header row")
    ap.add_argument("max_clear_rows", type=int)
    ap.add_argument("max_clear_cols", type=int)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--slice-start", type=int, default=0, help="0-based data-row offset")
    ap.add_argument("--slice-count", type=int, default=None)
    ap.add_argument("--skip-clear", action="store_true")
    args = ap.parse_args()

    full_data = pd.read_csv(Path(args.csv))
    stop = None if args.slice_count is None else args.slice_start + args.slice_count
    data = full_data.iloc[args.slice_start:stop].copy()
    values = data.astype(object).where(pd.notna(data), None).values.tolist()
    include_header = args.slice_start == 0
    matrix = ([list(full_data.columns)] if include_header else []) + values
    start_row = args.header_row - 1
    output_start_row = start_row if include_header else start_row + 1 + args.slice_start
    requests = []
    if include_header and not args.skip_clear:
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": args.sheet_id,
                    "startRowIndex": start_row,
                    "endRowIndex": args.max_clear_rows,
                    "startColumnIndex": 0,
                    "endColumnIndex": args.max_clear_cols,
                },
                "cell": {},
                "fields": "userEnteredValue",
            }
        })
    if include_header and len(full_data):
        requests.append({
            "copyPaste": {
                "source": {
                    "sheetId": args.sheet_id,
                    "startRowIndex": start_row + 1,
                    "endRowIndex": start_row + 2,
                    "startColumnIndex": 0,
                    "endColumnIndex": min(len(full_data.columns), args.max_clear_cols),
                },
                "destination": {
                    "sheetId": args.sheet_id,
                    "startRowIndex": start_row + 1,
                    "endRowIndex": start_row + 1 + len(full_data),
                    "startColumnIndex": 0,
                    "endColumnIndex": min(len(full_data.columns), args.max_clear_cols),
                },
                "pasteType": "PASTE_FORMAT",
                "pasteOrientation": "NORMAL",
            }
        })
    for offset in range(0, len(matrix), args.chunk):
        part = matrix[offset: offset + args.chunk]
        requests.append({
            "updateCells": {
                "range": {
                    "sheetId": args.sheet_id,
                    "startRowIndex": output_start_row + offset,
                    "endRowIndex": output_start_row + offset + len(part),
                    "startColumnIndex": 0,
                    "endColumnIndex": len(full_data.columns),
                },
                "rows": [{"values": [cell(v) for v in row]} for row in part],
                "fields": "userEnteredValue",
            }
        })
    print(json.dumps(requests, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
