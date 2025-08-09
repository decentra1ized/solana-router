import re
import json
from decimal import Decimal, getcontext
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

# ----- Settings -----
SQL_PATH = "pools.sql"
TOKENS_JSON_PATH = "tokens.json"

SOL  = "So11111111111111111111111111111111111111112"
USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# Set high precision for decimal calculations
getcontext().prec = 50

# ----- Utility: Load tokens.json -----
def load_token_map(tokens_json_path: str):
    addr_to_symbol = {}
    addr_to_decimals = {}
    p = Path(tokens_json_path)
    if not p.exists():
        return addr_to_symbol, addr_to_decimals
    data = json.loads(p.read_text(encoding="utf-8"))
    for item in data:
        addr = item.get("address")
        if not addr:
            continue
        addr_to_symbol[addr] = item.get("symbol") or addr
        dec = item.get("decimals")
        if dec is not None:
            addr_to_decimals[addr] = int(dec)
    return addr_to_symbol, addr_to_decimals

addr_to_symbol, addr_to_decimals = load_token_map(TOKENS_JSON_PATH)

def label(addr: str) -> str:
    return addr_to_symbol.get(addr, addr)

# ----- Parse pools.sql -----
insert_re = re.compile(
    r"INSERT\s+INTO\s+public\.pools\s*\((?P<cols>.*?)\)\s*VALUES\s*\((?P<vals>.*?)\);",
    re.IGNORECASE | re.DOTALL,
)

def split_csv_like(s: str):
    # Split by commas, but ignore commas inside single quotes
    return [p.strip() for p in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", s)]

def parse_sql_to_df(sql_path: str) -> pd.DataFrame:
    sql = Path(sql_path).read_text(encoding="utf-8")
    rows = []
    for m in insert_re.finditer(sql):
        cols_raw = m.group("cols")
        vals_raw = m.group("vals")
        cols = [c.strip().strip('"') for c in split_csv_like(cols_raw)]
        vals = [v.strip() for v in split_csv_like(vals_raw)]
        cleaned = []
        for v in vals:
            if v.upper() == "NULL":
                cleaned.append(None)
            else:
                if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
                    v = v[1:-1]
                cleaned.append(v)
        rows.append(dict(zip(cols, cleaned)))
    if not rows:
        raise RuntimeError("No INSERT data found in pools.sql.")
    df = pd.DataFrame(rows)
    # Convert types
    for c in ["tokenADecimals", "tokenBDecimals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["tokenABalance", "tokenBBalance", "fee", "protocolFee", "dynamicFeeAmount"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

df = parse_sql_to_df(SQL_PATH)

# ----- Pool model -----
@dataclass
class Pool:
    a: str
    b: str
    # Normalized reserves (taking 10^decimals into account)
    reserve_a: Decimal
    reserve_b: Decimal
    fee_total: Decimal  # 0~1
    raw: dict           # Original row

def safe_dec(x) -> Decimal:
    if x is None or x == "":
        return Decimal(0)
    return Decimal(str(x))

def normalize_amount(raw_amount: str, decimals: int | float | None) -> Decimal:
    d = int(decimals) if decimals is not None and str(decimals) != "nan" else None
    if d is None:
        # Fall back to raw amount if decimals are missing
        return safe_dec(raw_amount)
    return safe_dec(raw_amount) / (Decimal(10) ** d)

def build_pools(df: pd.DataFrame) -> list[Pool]:
    pools = []
    for _, row in df.iterrows():
        a = row["tokenA"]; b = row["tokenB"]
        a_dec = row.get("tokenADecimals")
        b_dec = row.get("tokenBDecimals")
        # Normalize reserves
        ra = normalize_amount(row.get("tokenABalance", "0"), a_dec)
        rb = normalize_amount(row.get("tokenBBalance", "0"), b_dec)
        # Sum up all fees
        fee = safe_dec(row.get("fee", "0"))
        pfee = safe_dec(row.get("protocolFee", "0"))
        dfee = safe_dec(row.get("dynamicFeeAmount", "0"))
        fee_total = fee + pfee + dfee
        # Clamp fee between 0 and 1
        if fee_total < 0: fee_total = Decimal(0)
        if fee_total > 1: fee_total = Decimal(1)
        pools.append(Pool(a=a, b=b, reserve_a=ra, reserve_b=rb, fee_total=fee_total, raw=row.to_dict()))
    return pools

pools = build_pools(df)

# For each token pair, select the pool with the highest liquidity (max k = x * y)
from collections import defaultdict
pair_to_best_pool_idx: dict[frozenset[str], int] = {}
pair_to_best_k: dict[frozenset[str], Decimal] = {}

for idx, p in enumerate(pools):
    pair_key = frozenset([p.a, p.b])
    k = p.reserve_a * p.reserve_b
    if pair_key not in pair_to_best_k or k > pair_to_best_k[pair_key]:
        pair_to_best_k[pair_key] = k
        pair_to_best_pool_idx[pair_key] = idx

def get_best_pool(x: str, y: str) -> Pool | None:
    idx = pair_to_best_pool_idx.get(frozenset([x, y]))
    return pools[idx] if idx is not None else None

# ----- Find qualified (A, B) pairs -----
def has_pair(x: str, y: str) -> bool:
    return frozenset([x, y]) in pair_to_best_pool_idx

tokens = set(df["tokenA"]).union(set(df["tokenB"]))
base_tokens = [t for t in tokens if t not in (SOL, USDC)]

qualified_pairs: list[tuple[str, str]] = []
for A in base_tokens:
    for B in base_tokens:
        if A >= B:
            continue
        if not (has_pair(A, B) and has_pair(A, SOL) and has_pair(SOL, B) and has_pair(A, USDC) and has_pair(USDC, B)):
            continue
        qualified_pairs.append((A, B))

# ----- AMM swap simulation (x*y = k, with fees) -----
def swap_out(pool: Pool, token_in: str, dx: Decimal) -> Decimal:
    """
    dx (normalized amount) swapped from token_in -> token_out
    Fee is applied: dx_eff = dx * (1 - fee)
    """
    if dx <= 0:
        return Decimal(0)
    fee = pool.fee_total
    dx_eff = dx * (Decimal(1) - fee)
    if token_in == pool.a:
        x, y = pool.reserve_a, pool.reserve_b
        dy = (dx_eff * y) / (x + dx_eff) if x + dx_eff > 0 else Decimal(0)
    elif token_in == pool.b:
        x, y = pool.reserve_b, pool.reserve_a
        dy = (dx_eff * y) / (x + dx_eff) if x + dx_eff > 0 else Decimal(0)
    else:
        raise ValueError("token_in not in pool")
    return dy

def simulate_path(path_tokens: list[str], amount_in: Decimal) -> Decimal | None:
    """
    path_tokens example: [A, B] or [A, SOL, B] or [A, USDC, B]
    amount_in: amount of A (normalized)
    """
    amt = amount_in
    for i in range(len(path_tokens) - 1):
        t_in = path_tokens[i]
        t_out = path_tokens[i+1]
        pool = get_best_pool(t_in, t_out)
        if not pool:
            return None
        out = swap_out(pool, token_in=t_in, dx=amt)
        amt = out
    return amt  # final amount of B

def avg_price(amount_in: Decimal, amount_out: Decimal) -> Decimal | None:
    if amount_out is None or amount_out == 0:
        return None
    # Average execution price: B per A
    return amount_out / amount_in

# ----- Main interaction -----
def main():
    if not qualified_pairs:
        print("No token pairs meet the 5 required conditions.")
        return

    # List qualified pairs
    print("Qualified token pairs:")
    for i, (A, B) in enumerate(qualified_pairs):
        print(f"[{i}] {label(A)} ({A})  <->  {label(B)} ({B})")

    # Get user selection
    sel = input("Select the number of the pair to analyze: ").strip()
    try:
        sel = int(sel)
        A, B = qualified_pairs[sel]
    except Exception:
        print("Invalid selection.")
        return

    amt_str = input(f"Enter the amount of {label(A)} to swap (e.g., 1.23): ").strip()
    try:
        amount_in = Decimal(amt_str)
        if amount_in <= 0:
            raise ValueError
    except Exception:
        print("Invalid amount.")
        return

    # Define paths
    path_direct = [A, B]
    path_sol    = [A, SOL, B]
    path_usdc   = [A, USDC, B]

    out_direct = simulate_path(path_direct, amount_in)
    out_sol    = simulate_path(path_sol, amount_in)
    out_usdc   = simulate_path(path_usdc, amount_in)

    # Average execution price: B per A
    px_direct = avg_price(amount_in, out_direct) if out_direct is not None else None
    px_sol    = avg_price(amount_in, out_sol) if out_sol is not None else None
    px_usdc   = avg_price(amount_in, out_usdc) if out_usdc is not None else None

    def fmt(x):
        return f"{x:.12f}" if x is not None else "-"

    print("\n=== Simulation Results (Unit: B per A) ===")
    print(f"1) Direct A-B     : out_B={fmt(out_direct)}  | avg(B/A)={fmt(px_direct)}")
    print(f"2) A-SOL-B (2 hop): out_B={fmt(out_sol)}     | avg(B/A)={fmt(px_sol)}")
    print(f"3) A-USDC-B (2hop): out_B={fmt(out_usdc)}    | avg(B/A)={fmt(px_usdc)}")

    # Choose the best route (max B output)
    candidates = [
        ("Direct A-B", out_direct, path_direct),
        ("A-SOL-B",    out_sol,    path_sol),
        ("A-USDC-B",   out_usdc,   path_usdc),
    ]
    viable = [(name, out, path) for (name, out, path) in candidates if out is not None]
    if not viable:
        print("\nNo viable routes found (insufficient liquidity or missing pools).")
        return
    best = max(viable, key=lambda x: x[1])
    best_name, best_out, best_path = best
    print(f"\n➡ Best Price Route (max B received): {best_name}  | Expected B received: {fmt(best_out)}")
    print("Route:", " -> ".join([label(t) for t in best_path]))

if __name__ == "__main__":
    main()
