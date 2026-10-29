# Finance notes

Quotes are MXN per USD. The company receives USD and converts it to MXN, so hedging means selling USD forward.

## Forward rate (covered interest parity, simple interest, Act/360)

```text
F = S * (1 + r_mxn * d / 360) / (1 + r_usd * d / 360)
forward points = F - S
```

`S` is the spot, `r` the annual rates (as fractions) and `d` the days to settlement.

## Proceeds of a partial hedge

```text
hedged USD     = notional * ratio
effective F    = F * (1 - spread_bps / 10000)
proceeds (MXN) = hedged * effective F + (notional - hedged) * realized spot - fee_usd * realized spot
```

The fixed fee is charged only when `ratio > 0`. The expected benefit is the proceeds minus `notional * forecast spot`; a negative value is the price of certainty.

## Historical simulation

- One hedge per month, placed on the first available day.
- Exposure is the USD cash-in of the next `horizon_days`, assumed known when hedging, so the study isolates FX risk.
- MXN costs are budgeted at `mxn_cost_share * exposure * spot at origin`.
- Strategies: no hedge, fixed ratio, and a forecast-guided ratio that hedges more when the forecast is below the forward.
- Uncertainty: 95 % percentile intervals from a moving-block bootstrap of the monthly margins.

## Commission avoided

The dashboard reports the bank cost avoided by hedging the selected ratio instead of 100 % of the amount. The assumptions (spread in basis points and a fixed fee) live in `configs/business.yaml`.

## Prediction intervals

They are empirical intervals based on the history of h-day log changes, not Value at Risk. A proper VaR needs backtest residuals and a stated confidence level and method; it is on the roadmap.
