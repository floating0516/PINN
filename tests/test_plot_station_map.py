from pathlib import Path

from scripts.plotting.plot_station_map import _inset_region, build_event_plot_plan


STATION_ROWS = {
    "Iquique 2014 M7.7": [
        {"event": "Iquique 2014 M7.7", "station_lat": -20.0, "station_lon": -70.0, "error": 0.2},
        {"event": "Iquique 2014 M7.7", "station_lat": -21.0, "station_lon": -69.0, "error": -0.1},
    ],
    "Nepal 2015 M7.3": [
        {"event": "Nepal 2015 M7.3", "station_lat": 27.6, "station_lon": 86.1, "error": 0.1},
    ],
}

EVENT_ROWS = {
    "Iquique 2014 M7.7": {
        "event": "Iquique 2014 M7.7",
        "mw_true": "7.7",
        "event_lat": "-20.5709",
        "event_lon": "-70.4931",
        "strike": "166.0",
        "dip": "75.0",
        "rake": "87.0",
    },
    "Nepal 2015 M7.3": {
        "event": "Nepal 2015 M7.3",
        "mw_true": "7.3",
        "event_lat": "27.8087",
        "event_lon": "86.0655",
        "strike": "102.0",
        "dip": "82.0",
        "rake": "87.0",
    },
}


def test_build_event_plot_plan_returns_one_output_per_event(tmp_path: Path) -> None:
    plans = build_event_plot_plan(
        stations_by_event=STATION_ROWS,
        events=EVENT_ROWS,
        output_dir=tmp_path,
    )

    assert [plan["event"] for plan in plans] == ["Iquique 2014 M7.7", "Nepal 2015 M7.3"]
    assert [plan["output_path"].name for plan in plans] == [
        "iquique_2014_m7_7.png",
        "nepal_2015_m7_3.png",
    ]


def test_build_event_plot_plan_uses_consistent_canvas_size(tmp_path: Path) -> None:
    plans = build_event_plot_plan(
        stations_by_event=STATION_ROWS,
        events=EVENT_ROWS,
        output_dir=tmp_path,
    )

    sizes = {(plan["figure_width_cm"], plan["figure_height_cm"]) for plan in plans}
    assert sizes == {(7.0, 6.0)}


def test_build_event_plot_plan_keeps_shared_color_limits(tmp_path: Path) -> None:
    plans = build_event_plot_plan(
        stations_by_event=STATION_ROWS,
        events=EVENT_ROWS,
        output_dir=tmp_path,
    )

    color_limits = {(plan["clim_min"], plan["clim_max"]) for plan in plans}
    assert color_limits == {(-0.2, 0.2)}


def test_build_event_plot_plan_uses_fixed_output_canvas(tmp_path: Path) -> None:
    plans = build_event_plot_plan(
        stations_by_event=STATION_ROWS,
        events=EVENT_ROWS,
        output_dir=tmp_path,
    )

    canvas_flags = {(plan["dpi"], plan["crop"]) for plan in plans}
    assert canvas_flags == {(300, True)}


def test_build_event_plot_plan_has_shared_target_pixel_size(tmp_path: Path) -> None:
    plans = build_event_plot_plan(
        stations_by_event=STATION_ROWS,
        events=EVENT_ROWS,
        output_dir=tmp_path,
    )

    pixel_sizes = {(plan["target_width_px"], plan["target_height_px"]) for plan in plans}
    assert pixel_sizes == {(900, 780)}


def test_build_event_plot_plan_exposes_reference_style_config(tmp_path: Path) -> None:
    plans = build_event_plot_plan(
        stations_by_event=STATION_ROWS,
        events=EVENT_ROWS,
        output_dir=tmp_path,
    )

    style_values = {
        (
            plan["triangle_size_cm"],
            plan["beachball_size_cm"],
            plan["north_arrow_cm"],
            plan["inset_width_cm"],
            plan["title_font"],
            plan["relief_grid"],
            plan["relief_cmap"],
            plan["land_fill"],
            plan["water_fill"],
            plan["relief_shading"],
            plan["inset_land_fill"],
            plan["inset_water_fill"],
            plan["inset_box_pen"],
            tuple(plan["frame_style"]),
            plan["scale_position"],
            plan["colorbar_size"],
            tuple(plan["colorbar_frame"]),
        )
        for plan in plans
    }
    assert style_values == {
        (
            0.22,
            0.5,
            0.78,
            1.55,
            "10p,Helvetica-Bold,black",
            Path.home() / ".gmt/server/earth/earth_relief/earth_relief_05m_g.grd",
            "grayC",
            "255/255/255",
            "194/226/247",
            "+a45+nt0.08",
            "247/243/223",
            "178/216/233",
            "0.55p,red",
            ("WSen", "xa1f1", "ya1f1"),
            "jBC",
            "1.7c/0.12c",
            ("xa0.2", "y+lM@-w@-"),
        )
    }


def test_inset_region_changes_with_main_region() -> None:
    asia_region = _inset_region([84, 89, 25, 29])
    south_america_region = _inset_region([-72, -68, -23, -18])

    assert asia_region != south_america_region
    assert asia_region[0] < 84 < asia_region[1]
    assert asia_region[2] < 25 < asia_region[3]
    assert south_america_region[0] < -72 < south_america_region[1]
    assert south_america_region[2] < -23 < south_america_region[3]
