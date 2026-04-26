"""
Build a 2-fold subject split for the Soli dataset.

Soli filenames follow:  <gesture>_<session>_<instance>.h5
For the cross-user setting, each unique session ID corresponds to one subject.
We collect those, sort them, and split 5/5 for fold 0 and swap for fold 1.

Usage:
    python scripts/make_splits.py --root data/dsp --out configs/splits.json
"""
import argparse, json, os, glob, re, sys


def parse_session(fname):
    # filenames look like '5_11_3.h5'  -> gesture=5, session=11, instance=3
    base = os.path.basename(fname)
    m = re.match(r"(\d+)_(\d+)_(\d+)\.h5", base)
    if m is None:
        return None
    g, s, i = m.groups()
    return int(g), int(s), int(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with .h5 files")
    ap.add_argument("--out", default="configs/splits.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*.h5")))
    if not files:
        print(f"no .h5 found in {args.root}", file=sys.stderr)
        sys.exit(1)

    sessions = set()
    parsed = []
    for f in files:
        p = parse_session(f)
        if p is None:
            continue
        parsed.append((f, p))
        sessions.add(p[1])

    sessions = sorted(sessions)
    print(f"found {len(parsed)} files across {len(sessions)} sessions: {sessions}")

    # map each session id -> subject index 0..N-1
    sess2subj = {s: i for i, s in enumerate(sessions)}
    n = len(sessions)
    half = n // 2

    fold0_train = sessions[:half]
    fold0_test  = sessions[half:]
    fold1_train = fold0_test
    fold1_test  = fold0_train

    splits = {
        "session_to_subject": {str(k): v for k, v in sess2subj.items()},
        "folds": [
            {"train_sessions": fold0_train, "test_sessions": fold0_test},
            {"train_sessions": fold1_train, "test_sessions": fold1_test},
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
