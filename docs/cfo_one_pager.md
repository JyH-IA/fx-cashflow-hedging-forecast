# Executive summary

**Scope.** Fictional operator with USD sales and MXN costs. Internal data (sales, cash-in) is synthetic. USD/MXN and the macro series are real (Banxico, FRED) up to 2026-10-30. Hedging rates and bank costs are illustrative assumptions. Not financial advice.

## Recommendation

The results do not support a specific hedge ratio. In the historical simulation a fixed hedge cut the variance of the monthly margin by about 16 % (95 % interval 2 % to 32 %), without a reliable gain in the mean margin or in the worst month. Read hedging here as a way to reduce volatility, not to earn more.

## What we expect

| Horizon | Expected USD/MXN | 95 % range | Expected USD inflow |
|---|---|---|---|
| 30 days | `<read from the dashboard>` | `<read from the dashboard>` | `<read from the dashboard>` |
| 60 days | `<read from the dashboard>` | `<read from the dashboard>` | `<read from the dashboard>` |
| 90 days | 18.80 (last observed rate, 18.7962) | about 17.7 to 21.3 | 11.6M USD (next 3 months) |

Source: the dashboard run of 2026-10-31. The expected rate is flat because the forecast repeats the last observed rate. The range is an empirical prediction interval, not Value at Risk.

## Does hedging help?

From `reports/hedging_backtest.md` (synthetic cash flows, 1000 block-bootstrap resamples):

- Margin variance falls 16 % with a fixed hedge (2 % to 32 %). The forecast-guided strategy also shows 16 %, but its interval (-4 % to 39 %) includes zero.
- The mean monthly margin rises by about 228K MXN (fixed) and 342K MXN (forecast), but both intervals include zero (-9K to 431K and -14K to 647K).
- The worst month does not improve: 6.25M MXN with no hedge, 6.25M fixed and 6.13M forecast-guided.
- The forecast-guided strategy used the random-walk expectation, because no model forecasts were provided. It does not measure forecast skill.

## What drives the forecast

Cash-in: the validation gate disabled the residual model, so the forecast is Prophet alone and SHAP does not apply. USD/MXN: the SHAP summary lists rolling averages of the consumer price index (38 %, 11 % and 10 % of the residual model's importance). That index moves in monthly steps and trends upward, so these features probably stand in for time. They should not be read as an economic driver of the exchange rate.

## How good is the model

From `reports/backtest.md` (five rolling origins). USD/MXN: no model clearly beats the random walk; ratios range from 0.92 to 1.00, and the hybrid equals it (1.000). Cash-in: the hybrid is best, at 0.60, 0.44 and 0.39 of the random walk's error at 30, 60 and 90 days, on synthetic seasonal data.

## Risks and limits

- Exchange rates are close to a random walk; forecasts can be wrong.
- Five backtest origins cannot tell small gaps from chance.
- Future interest rates, inflation and tourist arrivals are held at their last known value.
- The hedging study assumes the USD inflow is known when the hedge is placed.
- Bank spreads, fees and the interest rates of the forward are assumptions in `configs/business.yaml`.
