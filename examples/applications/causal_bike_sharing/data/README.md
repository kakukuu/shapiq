# Daily Bike Sharing data

`bike_sharing_daily.csv` contains the 731 daily observations from the UCI Bike
Sharing dataset's `day.csv` (2011–2012). It is distributed for reproducible use in
this example.

**Attribution:** Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning
Repository. [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894).
The [dataset page](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset)
licenses these data under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
This dataset license is separate from the repository's MIT code license.

The CSV retains the raw daily columns. The [local loader](../experiment_support/bike_data.py)
constructs the seven model inputs without changing the file:

| Input | Transformation |
|---|---|
| `trend` | Days elapsed since the first date, 2011-01-01 |
| `cosyear` | `cos(2 * pi * trend / 365)` |
| `sinyear` | `sin(2 * pi * trend / 365)` |
| `temp` | `47 * raw_temp - 8`, degrees Celsius |
| `atemp` | `66 * raw_atemp - 16`, degrees Celsius |
| `windspeed` | `67 * raw_windspeed`, original dataset scale |
| `hum` | `100 * raw_hum`, percent |

The target is the unchanged `cnt` daily rental count. Lag comparisons shift only
temperature, humidity and wind speed by one observation, then omit the first date.
