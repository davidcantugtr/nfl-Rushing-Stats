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
    args = ap.parse_args()

    data = pd.read_csv(Path(args.csv))
    matrix = [list(data.columns)] + data.astype(object).where(pd.notna(data), None).values.tolist()
    start_row = args.header_row - 1
    requests = [{
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
    }]
    if len(data):
        requests.append({
            "copyPaste": {
                "source": {
                    "sheetId": args.sheet_id,
                    "startRowIndex": start_row + 1,
                    "endRowIndex": start_row + 2,
                    "startColumnIndex": 0,
                    "endColumnIndex": min(len(data.columns), args.max_clear_cols),
                },
                "destination": {
                    "sheetId": args.sheet_id,
                    "startRowIndex": start_row + 1,
                    "endRowIndex": start_row + 1 + len(data),
                    "startColumnIndex": 0,
                    "endColumnIndex": min(len(data.columns), args.max_clear_cols),
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
                    "startRowIndex": start_row + offset,
                    "endRowIndex": start_row + offset + len(part),
                    "startColumnIndex": 0,
                    "endColumnIndex": len(data.columns),
                },
                "rows": [{"values": [cell(v) for v in row]} for row in part],
                "fields": "userEnteredValue",
            }
        })
    print(json.dumps(requests, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
