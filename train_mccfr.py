import pickle
from sweep import continue_train_mccfr
from mccfr import CFRTables
import os
import shutil

CHECKPOINT_PATH = "mccfr_checkpoint.pkl"
STARTING_STACK = 100_000
SEED = 42
CHECKPOINT_EVERY = 50_000
TARGET = 5_000_000

def save(tables, iters):
    # Save to main checkpoint
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"tables": tables, "iters": iters}, f)
    os.rename(tmp, CHECKPOINT_PATH)
    print(f"  [saved] {iters:,} iterations -> {CHECKPOINT_PATH}")
    
    # Backup every 500k iterations
    if iters % 500_000 == 0:
        backup = f"mccfr_checkpoint_{iters//1000}k.pkl"
        shutil.copy2(CHECKPOINT_PATH, backup)
        print(f"  [backup] -> {backup}")

def load():
    try:
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        print(f"  [loaded] resuming from {data['iters']:,} iterations")
        return data["tables"], data["iters"]
    except FileNotFoundError:
        return None, 0

tables, iters = load()
if tables is None:
    tables = CFRTables()

while iters < TARGET:
    next_checkpoint = min(iters + CHECKPOINT_EVERY, TARGET)
    tables = continue_train_mccfr(
        tables=tables,
        additional_iterations=next_checkpoint - iters,
        seed=SEED + iters,
        starting_stack=STARTING_STACK,
        desc=f"Train to {next_checkpoint:,}",
    )
    iters = next_checkpoint
    save(tables, iters)
    print(f"  infosets={len(tables.regret_sum):,}")

print(f"\nDone. {iters:,} iterations, {len(tables.regret_sum):,} infosets.")
