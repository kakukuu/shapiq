"""Human-readable input readings attached to individual-case figures.

The information band describes the observed sample and model inputs, never
attributions or synthetic imputer draws. Values are already in dataset units;
this module deliberately does not apply normalization or inverse scaling.
"""

from __future__ import annotations

import math
from datetime import date as calendar_date

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

FEATURES = ("trend", "cosyear", "sinyear", "temp", "atemp", "windspeed", "hum")
LAGGED_FEATURES = ("temp", "hum", "windspeed")
SCHEMA_VERSION = 1


def _number(value):
    """Convert NumPy scalars and missing pandas values into JSON-safe numbers."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _date(value):
    if value is None or str(value) in ("", "NaT", "nan", "<NA>"):
        return "n/a"
    return str(value).split("T")[0].split(" ")[0]


def _format(value, digits=1):
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _weather(values):
    return (
        f"Temperature (temp) {_format(values['temp'], 2)} °C | "
        f"Feels-like (atemp) {_format(values['atemp'], 2)} °C | "
        f"Humidity (hum) {_format(values['hum'], 2)}% | "
        f"Wind speed (windspeed) {_format(values['windspeed'], 2)} (dataset scale)"
    )


def build_sample_readings(point, arm):
    """Describe one point-table row, including the paired Lag input design.

    ``point`` accepts a dict or pandas Series with the seven original feature
    columns, ``date``, ``row_id``, ``y_true``, and both prediction columns.
    Lag rows additionally carry ``*_lag_design`` and ``lag_source_date``.
    Missing readings remain ``None`` in the returned JSON-safe dict and show
    as ``n/a``. Pass ``None`` for an aggregate figure; no sample is invented.
    """
    if point is None:
        return None
    arm = str(arm)
    is_lag = arm.lower() == "lag"
    observed = {feature: _number(point.get(feature)) for feature in FEATURES}
    baseline = observed.copy()
    modified = observed.copy()
    if is_lag:
        modified = {
            feature: _number(
                point.get(
                    f"{feature}_lag_design",
                    None if feature in LAGGED_FEATURES else observed[feature],
                )
            )
            for feature in FEATURES
        }
    date = _date(point.get("date"))
    try:
        weekday = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")[
            calendar_date.fromisoformat(date).weekday()
        ]
    except ValueError:
        weekday = None
    lag_source_date = _date(point.get("lag_source_date")) if is_lag else None
    row_id = _number(point.get("row_id"))
    if row_id is not None and row_id.is_integer():
        row_id = int(row_id)
    outcomes = {
        name: _number(point.get(name))
        for name in ("y_true", "prediction_base", "prediction_variant")
    }
    lines = []
    if is_lag:
        lines.append(f"Baseline / same-day daily mean weather ({date}): {_weather(baseline)}")
        lines.append(
            f"Lag model inputs: temp {_format(modified['temp'], 2)} °C, "
            f"hum {_format(modified['hum'], 2)}%, "
            f"windspeed {_format(modified['windspeed'], 2)} (dataset scale) "
            f"from previous day {lag_source_date}; "
            f"atemp {_format(modified['atemp'], 2)} °C remains same-day."
        )
    else:
        lines.append(f"Daily mean weather / same input in both arms: {_weather(observed)}")
    lines.append(
        f"Other model inputs: trend {_format(observed['trend'], 0)} (days since 2011-01-01) | "
        f"cosyear {_format(observed['cosyear'], 3)} | "
        f"sinyear {_format(observed['sinyear'], 3)} (season encodings, dimensionless)"
    )
    if is_lag:
        prediction = (
            f"baseline prediction {_format(outcomes['prediction_base'])} | "
            f"Lag prediction {_format(outcomes['prediction_variant'])}"
        )
    else:
        # Display both values rather than asserting equality for incomplete or
        # unexpected inputs. Ordinary A/B arms use the same prediction model.
        prediction = (
            f"baseline prediction {_format(outcomes['prediction_base'])} | "
            f"modified prediction {_format(outcomes['prediction_variant'])}"
        )
    lines.append(
        f"Outcome / predictions (not model inputs; rentals/day): "
        f"observed {_format(outcomes['y_true'], 0)} | {prediction}"
    )
    return {
        "schema": SCHEMA_VERSION,
        "title": "Raw readings / model inputs, not attributions",
        "date": date,
        "weekday": weekday,
        "row_id": row_id,
        "arm": arm,
        "is_lag": is_lag,
        "lag_source_date": lag_source_date,
        "lines": lines,
        "observed_raw_values": observed,
        "input_raw_values": {"baseline": baseline, "modified": modified},
        "outcomes": outcomes,
        "units": {
            "temp": "°C",
            "atemp": "°C",
            "hum": "%",
            "windspeed": "dataset scale",
            "trend": "days since 2011-01-01",
            "cosyear": "dimensionless",
            "sinyear": "dimensionless",
            "y_true": "daily rentals",
            "prediction_base": "daily rentals",
            "prediction_variant": "daily rentals",
        },
    }


def _wrap_line(line, renderer, font, width_px):
    """Wrap with rendered font widths instead of a character-count heuristic."""
    output, current = [], ""
    for word in str(line).split():
        candidate = f"{current} {word}" if current else word
        width = renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
        if current and width > width_px:
            output.append(current)
            current = word
        else:
            current = candidate
    if current:
        output.append(current)
    return output or [""]


def add_sample_readings(fig, readings):
    """Add a top information band without shrinking an existing plot.

    Axes, figure titles, footer text and figure legends retain their physical
    locations measured from the bottom of the original figure. The canvas
    grows upward to accommodate wrapped readings at a fixed readable font.
    Add this after the plot's layout is complete and before saving/displaying.
    Repeated attachment is a no-op, preventing duplicate headers.
    """
    if readings is None or getattr(fig, "sample_readings", None) is not None:
        return fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = (float(value) for value in fig.get_size_inches())
    content_font = FontProperties(size=9.5)
    title_font = FontProperties(size=10.5, weight="bold")
    available_width = width * fig.dpi * 0.93
    title = readings["title"]
    if readings.get("date", "n/a") != "n/a":
        title += f" | {readings['date']}"
        if readings.get("weekday"):
            title += f" ({readings['weekday']})"
    if readings.get("row_id") is not None:
        title += f" | row {readings['row_id']}"
    title_lines = _wrap_line(title, renderer, title_font, available_width)
    lines = [
        part
        for line in readings["lines"]
        for part in _wrap_line(line, renderer, content_font, available_width)
    ]
    line_height = 0.195
    band_height = 0.22 + 0.23 * len(title_lines) + line_height * len(lines)
    added_height = band_height + 0.16
    new_height = height + added_height
    ratio = height / new_height
    axes_positions = [(ax, ax.get_position(original=True).frozen()) for ax in fig.axes]
    text_positions = [
        (artist, artist.get_position())
        for artist in fig.texts
        if artist.get_transform() is fig.transFigure
    ]
    legend_positions = [
        (legend, legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds)
        for legend in fig.legends
    ]
    # A later constrained-layout pass must not stretch the original panels
    # into the newly added information area.
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    fig.set_size_inches(width, new_height, forward=True)
    for ax, bounds in axes_positions:
        ax.set_position([bounds.x0, bounds.y0 * ratio, bounds.width, bounds.height * ratio])
    for artist, (x, y) in text_positions:
        artist.set_position((x, y * ratio))
    for legend, (x, y, bbox_width, bbox_height) in legend_positions:
        legend.set_bbox_to_anchor(
            (x, y * ratio, bbox_width, bbox_height * ratio), transform=fig.transFigure
        )
    band_bottom = height + 0.08
    background = Rectangle(
        (0.02, band_bottom / new_height),
        0.96,
        band_height / new_height,
        transform=fig.transFigure,
        facecolor="#f2f4f6",
        edgecolor="#d8dfe5",
        linewidth=0.7,
        zorder=-1,
        clip_on=False,
    )
    background.set_gid("sample-readings-background")
    fig.add_artist(background)
    y = band_bottom + band_height - 0.12
    texts = []
    for line in title_lines:
        texts.append(
            fig.text(
                0.035,
                y / new_height,
                line,
                fontproperties=title_font,
                color="#25313d",
                va="top",
                ha="left",
            )
        )
        y -= 0.23
    for line in lines:
        texts.append(
            fig.text(
                0.035,
                y / new_height,
                line,
                fontproperties=content_font,
                color="#394653",
                va="top",
                ha="left",
            )
        )
        y -= line_height
    for artist in texts:
        artist.set_gid("sample-readings-text")
    fig.sample_readings = readings
    fig.sample_readings_artists = texts
    fig.sample_readings_band = background
    return fig
