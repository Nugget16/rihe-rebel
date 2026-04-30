from __future__ import annotations

import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rihe_engine import RIHEEngine
from mccfr import AvgStrategyPolicy, CFRTables
from belief import BeliefState
from pbs import encode_pbs
from value_net import ValueNet, get_device


def generate_trajectory(
    engine: RIHEEngine,
    player: int,
    policy: AvgStrategyPolicy,
    starting_stack: int,
    rng: random.Random,
) -> list[tuple[np.ndarray, float]]:
    """
    Play one hand using `policy` for both players.
    Returns list of (pbs_vector, terminal_value) pairs for `player`.
    """
    engine.reset(starting_stack)
    engine.deal_hand()

    belief = BeliefState()
    my_card = engine.h.hole[player]
    belief.reset(my_card, [])

    samples = []
    start_stack = engine.stacks[player]

    while not engine.h.terminal:
        if engine.is_chance_node():
            outcomes = engine.chance_outcomes()
            card = outcomes[rng.randrange(len(outcomes))]
            engine.apply_chance(card)
            # Update belief when community card is revealed
            belief.observe_card(card)
            continue

        to_act = engine.current_player()

        # Record PBS at player's decision points
        if to_act == player:
            pbs = encode_pbs(engine, player, belief, starting_stack)
            samples.append(pbs)

        # Sample action from policy
        action = policy.sample_action(engine, to_act)

        # Update belief when opponent acts
        if to_act != player:
            belief.update(engine, to_act, action, policy)

        engine.apply(action)

    # Terminal value for player
    terminal_value = float(engine.stacks[player] - start_stack)

    # Assign terminal value to all decision points in this hand
    return [(pbs, terminal_value) for pbs in samples]


def collect_trajectories(
    n_hands: int,
    policy: AvgStrategyPolicy,
    starting_stack: int = 100_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect PBS vectors and terminal values from n_hands of self-play.
    Returns (X, y) arrays for training.
    """
    rng = random.Random(seed)
    all_pbs = []
    all_vals = []

    for _ in tqdm(range(n_hands), desc="Collecting trajectories"):
        eng = RIHEEngine(random.Random(rng.randrange(1 << 30)))
        for player in (0, 1):
            samples = generate_trajectory(eng, player, policy, starting_stack, rng)
            for pbs, val in samples:
                all_pbs.append(pbs)
                all_vals.append(val)

    X = np.array(all_pbs, dtype=np.float32)
    y = np.array(all_vals, dtype=np.float32)
    return X, y


# Training

def train_value_net(
    net: ValueNet,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_split: float = 0.1,
) -> list[float]:
    """
    Train the value network on collected trajectories.
    Returns list of validation losses per epoch.
    """
    n = len(X)
    n_val = int(n * val_split)
    idx = np.random.permutation(n)
    X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
    X_tr,  y_tr  = X[idx[n_val:]], y[idx[n_val:]]

    X_tr_t  = torch.tensor(X_tr,  device=device)
    y_tr_t  = torch.tensor(y_tr,  device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device).unsqueeze(1)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    net.to(device)

    val_losses = []
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        net.eval()
        with torch.no_grad():
            val_pred = net(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        print(f"  Epoch {epoch}/{epochs} | train_loss={total_loss/len(X_tr):.5f} | val_loss={val_loss:.5f}")

    return val_losses


# Main training loop

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="mccfr_checkpoint.pkl")
    parser.add_argument("--net_out",     default="value_net.pt")
    parser.add_argument("--n_hands",     type=int, default=50_000)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load MCCFR tables
    print(f"Loading MCCFR checkpoint from {args.checkpoint}...")
    with open(args.checkpoint, "rb") as f:
        data = pickle.load(f)
    tables: CFRTables = data["tables"]
    print(f"  {data['iters']:,} iterations, {len(tables.regret_sum):,} infosets")

    policy = AvgStrategyPolicy(tables, rng=random.Random(args.seed))

    # Collect trajectories
    print(f"\nCollecting trajectories from {args.n_hands:,} hands...")
    X, y = collect_trajectories(
        n_hands=args.n_hands,
        policy=policy,
        seed=args.seed,
    )
    print(f"  Collected {len(X):,} PBS samples")
    print(f"  Value range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Non-zero values: {(y != 0).sum()} / {len(y)}")
    print(f"  Std dev: {y.std():.4f}")

    # Train
    print(f"\nTraining value network ({args.epochs} epochs)...")
    net = ValueNet()
    val_losses = train_value_net(
        net=net,
        X=X,
        y=y,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save
    net.save(args.net_out)
    print(f"\nSaved value network to {args.net_out}")
    print(f"Final val loss: {val_losses[-1]:.5f}")


if __name__ == "__main__":
    main()
