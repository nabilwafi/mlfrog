# Triple-Barrier Report
- sl_mult (ATR): 1.5
- tp_mult (ATR): 2.0
- horizon_bars: 8

## LONG label distribution
- win (1): 16458 (24.11%)
- loss (0): 25055 (36.70%)
- timeout (-1): 26758 (39.19%)
- win_rate_train (win/(win+loss)): 0.3965
- avg/median bars_to_resolution (win): 4.460 / 4.000
- avg/median bars_to_resolution (loss): 3.941 / 4.000

## SHORT label distribution
- win (1): 16346 (23.94%)
- loss (0): 25446 (37.27%)
- timeout (-1): 26479 (38.79%)
- win_rate_train (win/(win+loss)): 0.3911
- avg/median bars_to_resolution (win): 4.299 / 4.000
- avg/median bars_to_resolution (loss): 4.043 / 4.000

## Breakeven check (implied win-rate)
- min_winrate = SL / (SL + TP) = 0.4286
- LONG win_rate_train=0.3965 -> BELOW (needs better params)
- SHORT win_rate_train=0.3911 -> BELOW (needs better params)

## Session breakdown (pct of labels)
| session | LONG win% | LONG loss% | LONG timeout% | SHORT win% | SHORT loss% | SHORT timeout% |
|---|---:|---:|---:|---:|---:|---:|
| asia | 18.78% | 29.65% | 51.56% | 17.97% | 31.03% | 51.00% |
| london | 33.78% | 48.97% | 17.25% | 34.19% | 48.54% | 17.28% |
| london_ny_overlap | 33.61% | 50.97% | 15.41% | 35.43% | 48.98% | 15.59% |
| ny | 19.75% | 30.45% | 49.80% | 19.17% | 31.90% | 48.93% |