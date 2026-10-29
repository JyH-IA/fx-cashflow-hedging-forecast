# Executive summary (template)

> **How to use.** Run `make demo` (or the real-data pipeline), then copy the figures from `reports/` into the fields marked `<...>`. Keep it under 350 words and delete this note. The values below are intentionally left blank: this repository never ships invented results.

**Scope.** Fictional operator with USD sales and MXN costs. Synthetic internal data. Not financial advice.

## Recommendation

`<One sentence: for example, hedge between X % and Y % of the next 30 days of USD inflow.>`

## What we expect

| Horizon | Expected USD/MXN | 95 % range | Expected USD inflow |
|---|---|---|---|
| 30 days | `<...>` | `<...>` | `<...>` |
| 60 days | `<...>` | `<...>` | `<...>` |
| 90 days | `<...>` | `<...>` | `<...>` |

Source: `GET /forecast` or the dashboard. The range is an empirical prediction interval, not Value at Risk.

## Does hedging help?

`<From reports/hedging_backtest.md: change in margin variance with its interval, mean monthly margin difference and total hedge cost.>`

## What drives the forecast

`<From reports/shap_drivers.md: the three main drivers in plain language.>`

## How good is the model

`<From reports/backtest.md: rmse_vs_random_walk by horizon. If it is not below 1, say so plainly: the model does not beat a random walk.>`

## Risks and limits

- Exchange rates are close to a random walk; forecasts can be wrong.
- Future interest rates, inflation and tourist arrivals are held at their last known value.
- The hedging study assumes the USD inflow is known when the hedge is placed.
- Bank spreads and fees are assumptions in `configs/business.yaml`.
