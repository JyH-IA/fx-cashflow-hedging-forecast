# Executive summary

**Scope.** Fictional operator with USD sales and MXN costs. Internal data (sales, cash-in) is synthetic. USD/MXN and the macro series are real (Banxico, FRED) up to 2026-10-30. Hedging rates and bank costs are illustrative assumptions. Not financial advice.

## Recommendation

The results do not support a specific hedge ratio. In the historical simulation a fixed hedge cut the variance of the monthly margin by about 18 % (95 % interval 4 % to 34 %), without a reliable gain in the mean margin or in the worst month. Read hedging here as a way to reduce volatility, not to earn more.

## What we expect

| Horizon | Expected USD/MXN | 95 % range | Expected USD inflow |
|---|---|---|---|
| 30 days | 18.80 | 18.09 to 20.39 | 3.5M USD (next month) |
| 60 days | 18.80 | 17.78 to 20.76 | 7.4M USD (next 2 months) |
| 90 days | 18.80 | 17.65 to 21.29 | 11.7M USD (next 3 months) |

Source: the dashboard run of 2026-10-31. The expected rate is flat at the last observed rate (18.7962 on 2026-10-30) because the forecast repeats it. The range is an empirical prediction interval, not Value at Risk.

## Does hedging help?

From `reports/hedging_backtest.md` (synthetic cash flows, 1000 block-bootstrap resamples):

- Margin variance falls 18 % with a fixed hedge (4 % to 34 %). The forecast-guided strategy shows 19 %, but its interval (-1 % to 41 %) includes zero.
- The mean monthly margin rises by about 146K MXN (fixed) and 219K MXN (forecast-guided), but both intervals include zero (-92K to 348K and -138K to 522K).
- The worst month barely changes: 6.19M MXN with no hedge, 6.24M fixed and 6.10M forecast-guided.
- The forecast-guided strategy used the random-walk expectation, because no model forecasts were provided. It does not measure forecast skill.
- The forward was priced with 6.22 % for Mexico and 3.88 % for the US (`configs/business.yaml`, illustrative), so the size of any gain in the mean margin depends on those rates.

## What drives the forecast

Neither forecast has a residual model: the validation gate rejected it for both targets. USD/MXN repeats the last observed rate, so nothing drives it. Cash-in follows Prophet's trend and seasonality, which come from the seasonal pattern built into the synthetic data. SHAP does not apply to either target.

## How good is the model

From `reports/backtest.md` (five rolling origins). USD/MXN: no model clearly beats the random walk; ratios range from 0.92 to 1.00, and the hybrid equals it (1.000). Cash-in: the hybrid is best, at 0.57, 0.43 and 0.38 of the random walk's error at 30, 60 and 90 days, on synthetic seasonal data.

## Risks and limits

- Exchange rates are close to a random walk; forecasts can be wrong.
- Five backtest origins cannot tell small gaps from chance.
- Future interest rates, inflation and tourist arrivals are held at their last known value.
- The hedging study assumes the USD inflow is known when the hedge is placed.
- Bank spreads, fees and the interest rates of the forward are assumptions in `configs/business.yaml`.
