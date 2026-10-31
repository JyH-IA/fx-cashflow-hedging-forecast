# Finance notes

Quotes are MXN per USD. The company receives USD and converts it to MXN, so hedging means selling USD forward.

## Forward rate (covered interest parity, simple interest, Act/360)

```text
F = S * (1 + r_mxn * d / 360) / (1 + r_usd * d / 360)
forward points = F - S
```

`S` is the spot, `r` the annual rates (as fractions) and `d` the days to settlement. The rates come from `configs/business.yaml` (`rate_mxn_pct`, `rate_usd_pct`). They are illustrative inputs, so the gap between the forward and the spot shown in the dashboard is that rate gap, not a prediction.

## Proceeds of a partial hedge

```text
hedged USD     = notional * ratio
effective F    = F * (1 - spread_bps / 10000)
proceeds (MXN) = hedged * effective F + (notional - hedged) * realized spot - fee_usd * realized spot
```

The fixed fee is charged only when `ratio > 0`. The expected benefit is the proceeds minus `notional * forecast spot`; a negative value is the price of certainty.

## Figures in the dashboard

- **Bad case and good case:** the low and high ends of the 95 % historical range of the move over the horizon, applied to the settlement-day rate.
- **Downside protected in a bad case:** the bad-case proceeds of the selected hedge minus the bad-case proceeds without a hedge.
- **Swing narrowed by:** how much the gap between the good and the bad case shrinks compared with no hedge.
- **Bank cost of this hedge:** the spread and the fixed fee of `configs/business.yaml` for the selected ratio.

The dashboard does not show the expected benefit: with a flat forecast it only reflects the forward premium and could be read as a gain.

## Historical simulation

- One hedge per month, placed on the first available day.
- Exposure is the USD cash-in of the next `horizon_days`, assumed known when hedging, so the study isolates FX risk.
- MXN costs are budgeted at `mxn_cost_share * exposure * spot at origin`.
- Strategies: no hedge, fixed ratio, and a forecast-guided ratio that hedges more when the forecast is below the forward.
- Uncertainty: 95 % percentile intervals from a moving-block bootstrap of the monthly margins.

## Commission avoided

`commission_saving_mxn` in `fxhedge.dashboard` computes the bank cost avoided by hedging the selected ratio instead of 100 %. The dashboard does not show it since its redesign.

## Prediction intervals

They are empirical intervals based on the history of h-day log changes, not Value at Risk. A proper VaR needs backtest residuals and a stated confidence level and method; it is on the roadmap.
