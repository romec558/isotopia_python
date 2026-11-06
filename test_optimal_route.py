"""Tests for scripts/optimal_route.py.

The first group needs no data. The last test needs ISOTOPIA_DIR and the proton
library for Mo-100 (skipped otherwise).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("optimal_route", ROOT / "scripts" / "optimal_route.py")
opt = importlib.util.module_from_spec(_spec)
sys.modules["optimal_route"] = opt          # dataclasses need the module registered
_spec.loader.exec_module(opt)


# ---------------------------------------------------------------- names and lists

def test_parse_and_display_names():
    assert opt.parse_nuclide("Tc99_m1") == (43, 99, 1)
    assert opt.parse_nuclide("Tc99_g") == (43, 99, 0)
    assert opt.parse_nuclide("Tc99") == (43, 99, None)
    assert opt.nuclide_name(43, 99, 1) == "Tc99_m1"
    assert opt.nuclide_name(43, 99, None) == "Tc99"
    for bad in ("tc99", "Tc099m", "Xx99", "Tc99_m3"):
        with pytest.raises(ValueError):
            opt.parse_nuclide(bad)


def test_read_nuclide_list(tmp_path):
    f = tmp_path / "exclude.txt"
    f.write_text("# blacklist\nMo99   # comment\nTc99_m1, Ru99_g\n\n")
    assert opt.read_nuclide_list(f) == {(42, 99, None), (43, 99, 1), (44, 99, 0)}
    f.write_text("Mo99\nnot_a_nuclide\n")
    with pytest.raises(ValueError, match="line 2"):
        opt.read_nuclide_list(f)


def test_listed_matching_rules():
    entries = {(42, 99, None), (43, 99, 1)}
    assert opt.listed(42, 99, 0, entries)            # bare entry matches every state
    assert opt.listed(42, 99, 1, entries)
    assert opt.listed(43, 99, 1, entries)
    assert not opt.listed(43, 99, 0, entries)        # Tc99_m1 does not match the ground state
    assert opt.listed(43, 99, -1, entries)           # ISOTOPIA 'total' matches any listed state
    assert not opt.listed(44, 99, -1, entries)


def test_energy_grid():
    g = opt.energy_grid(0.001, 200.0, 1.0)
    assert g[0] == pytest.approx(0.001) and g[-1] == pytest.approx(199.001) and len(g) == 200
    assert opt.energy_grid(1.0, 3.0, 0.5).tolist() == [1.0, 1.5, 2.0, 2.5, 3.0]


# ---------------------------------------------------------------- Eback solver

def test_eback_analytic(monkeypatch):
    """With S(E) = k/E the range integral is (Ebeam^2 - Eback^2) / (2k), so
    Eback = sqrt(Ebeam^2 - 2 k d) exactly."""
    k = 200.0                                         # MeV^2 / cm
    monkeypatch.setattr(opt, "stoppingpower", lambda st, E: np.float32(k / E))
    solver = opt.EbackSolver(2, 42, 100, npts=20000)
    for ebeam, d in ((20.0, 0.05), (50.0, 1.0), (100.0, 5.0)):
        eback, stopped = solver.eback(ebeam, d)
        assert not stopped
        assert eback == pytest.approx(np.sqrt(ebeam**2 - 2 * k * d), rel=1e-4)
    # Range shorter than the target: beam stops, Eback clipped to the floor
    eback, stopped = solver.eback(5.0, 10.0)
    assert stopped and eback == solver.E_floor


def test_eback_floor_is_stopping_power_maximum():
    """The real Bethe-Bloch floor for protons on Mo sits near 0.5 MeV and S > 0 above it."""
    solver = opt.EbackSolver(2, 42, 100)
    assert 0.1 < solver.E_floor < 1.0
    assert np.all(np.diff(solver.R) > 0)              # range increases monotonically


# ---------------------------------------------------------------- with the library

ISOTOPIA = Path(os.environ.get("ISOTOPIA_DIR") or ROOT.parent / "isotopia")
needs_lib = pytest.mark.skipif(not (ISOTOPIA / "files").is_dir(), reason="ISOTOPIA_DIR not set")


@needs_lib
def test_mo100_route_and_precursor_exclusion():
    """p + Mo-100 at 20 MeV makes Tc-99m; Mo-99 feeds it by beta- decay in the
    model, so listing Mo99 must exclude the route. ISOTOPIA's own Leff must match
    the requested thickness."""
    st_paths = opt.State()
    opt.machine(st_paths)
    t = opt.Target(42, 100, True)
    solver = opt.EbackSolver(2, 42, 100)
    eback, _ = solver.eback(20.0, 0.05)
    lines = ["projectile p", "element Mo", "mass 100", "Ibeam 1", "Tirrad 1 d", "Tcool 1 d",
             "Ebeam 20.0", f"Eback {eback:.6f}"]
    di = (43, 99, 1)
    row = opt.run_one(opt.RunSpec(di, 2, t, 20.0, eback, lines))
    assert row["status"] == "ok" and row["A_EOC_Bq"] > 0
    assert row["Leff_cm"] == pytest.approx(0.05, rel=1e-3)
    row = opt.run_one(opt.RunSpec(di, 2, t, 20.0, eback, lines, {(42, 99, None)}))
    assert row["status"] == "excluded" and "Mo99" in row["note"]
