import argparse
import json


def fit_alpha(real, estimated):
    real, estimated = _check(real, estimated)
    s_xy = sum(x * y for x, y in zip(estimated, real))
    s_xx = sum(x * x for x in estimated)
    if s_xx == 0:
        raise ValueError("all estimated values are zero")
    return s_xy / s_xx


def fit_alpha_intercept(real, estimated):
    real, estimated = _check(real, estimated)
    n = len(real)
    sx = sum(estimated)
    sy = sum(real)
    sxx = sum(x * x for x in estimated)
    sxy = sum(x * y for x, y in zip(estimated, real))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("estimated has zero variance, cannot fit line with intercept")
    alpha = (n * sxy - sx * sy) / denom
    b = (sy - alpha * sx) / n
    return alpha, b


def _check(real, estimated):
    if len(real) != len(estimated):
        raise ValueError(f"real({len(real)}) and estimated({len(estimated)}) have different lengths")
    if len(real) == 0:
        raise ValueError("input lists are empty")
    return list(real), list(estimated)


def mape(real, estimated, alpha, b=0.0):
    errs = [abs(y - (alpha * x + b)) / abs(y) for x, y in zip(estimated, real) if y != 0]
    return 100.0 * sum(errs) / len(errs) if errs else float("nan")


def r2(real, estimated, alpha, b=0.0):
    ybar = sum(real) / len(real)
    ss_res = sum((y - (alpha * x + b)) ** 2 for x, y in zip(estimated, real))
    ss_tot = sum((y - ybar) ** 2 for y in real)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _parse_list(s):
    return [float(x) for x in s.replace("，", ",").split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Fit alpha in the linear model: real = alpha * estimated (+ b if --intercept).")
    ap.add_argument("--real", type=str, help="List of real index storage values, comma-separated")
    ap.add_argument("--est", type=str, help="List of estimated index storage values, comma-separated")
    ap.add_argument("--json", type=str, help="JSON file containing 'real' and 'estimated' lists")
    ap.add_argument("--intercept", action="store_true", help="Fit a line with intercept: real = alpha*est + b")
    args = ap.parse_args()

    if args.json:
        d = json.load(open(args.json, encoding="utf-8"))
        real = d["real"]
        estimated = d["estimated"]
    elif args.real and args.est:
        real = _parse_list(args.real)
        estimated = _parse_list(args.est)
    else:
        ap.error("需提供 --real 和 --est，或 --json 文件")

    real, estimated = _check(real, estimated)
    n = len(real)

    if args.intercept:
        alpha, b = fit_alpha_intercept(real, estimated)
        print(f"model: real = alpha * estimated + b")
        print(f"alpha = {alpha:.6f}")
        print(f"b     = {b:.6f}")
        print(f"MAPE  = {mape(real, estimated, alpha, b):.2f}%")
        print(f"R^2   = {r2(real, estimated, alpha, b):.4f}")
    else:
        alpha = fit_alpha(real, estimated)
        print(f"model: real = alpha * estimated (through origin)")
        print(f"alpha = {alpha:.6f}")
        print(f"MAPE  = {mape(real, estimated, alpha):.2f}%")
        print(f"R^2   = {r2(real, estimated, alpha):.4f}")

    print(f"\n{n} samples:")
    print(f"{'estimated':>12}{'real':>12}{'ratio(real/est)':>16}")
    for x, y in zip(estimated, real):
        ratio = y / x if x != 0 else float("nan")
        print(f"{x:>12.4f}{y:>12.4f}{ratio:>16.4f}")


if __name__ == "__main__":
    main()
