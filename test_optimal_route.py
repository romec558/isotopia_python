"""Tests for scripts/optimal_route.py.

The first group needs no data. The last group needs ISOTOPIA_DIR and the proton
library for Mo-100 and Ru-101 (skipped otherwise).
"""
from __future__ import annotations

import importlib.util
import io
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


def test_candidate_targets_mass_window(tmp_path):
    """Targets span A-5..A+5; lighter ones that cannot reach A_DI (A + 1 < A_DI
    for protons) are skipped with a warning, isomeric targets are skipped."""
    lib = tmp_path / "libs"
    for name in ("Mo094", "Mo098", "Mo100", "Ru104", "Ru105", "Tc099m"):
        (lib / "p" / name).mkdir(parents=True)
    ab = tmp_path / "code" / "files" / "abundance"
    ab.mkdir(parents=True)
    (ab / "z042").write_text("".join(f"{42:4d}{a:4d}{10.0:11.6f}\n" for a in (94, 98, 100)))
    (ab / "z044").write_text(f"{44:4d}{104:4d}{10.0:11.6f}\n")
    warnings: list[str] = []
    targets = opt.candidate_targets((43, 99, 1), 2, str(tmp_path / "code"), str(lib), False, warnings)
    assert [t.name for t in targets] == ["Mo98", "Mo100", "Ru104"]
    assert any("Mo094" in w and "cannot reach" in w for w in warnings)
    assert any("Tc099m" in w for w in warnings)
    # The DI's own ground state is not a production route; its isomer from it is
    warnings.clear()
    (lib / "p" / "Tc099").mkdir()
    (ab / "z043").write_text(f"{43:4d}{99:4d}{10.0:11.6f}\n")
    names = [t.name for t in opt.candidate_targets((43, 99, 0), 2, str(tmp_path / "code"), str(lib),
                                                   False, warnings)]
    assert "Tc99" not in names and any("is the DI itself" in w for w in warnings)
    names = [t.name for t in opt.candidate_targets((43, 99, 1), 2, str(tmp_path / "code"), str(lib),
                                                   False, [])]
    assert "Tc99" in names


# ---------------------------------------------------------------- ranking metrics

def _row(target, ebeam, rate, eoi, eoc):
    return {"projectile": "p", "target": target, "stable": True, "Ebeam_MeV": ebeam,
            "Eback_MeV": ebeam - 5, "Leff_cm": 0.05, "m_rate": rate, "m_rate_unit": "g/(mA h)",
            "m_EOI_g": eoi, "m_EOC_g": eoc, "status": "ok", "note": ""}


def test_metric_choice_changes_ranking(tmp_path):
    """Route A wins on initial rate and EOI mass, route B on end-of-cooling mass;
    the summary names the winner of the chosen metric and lists all three."""
    rows = [_row("Aa10", 20.0, 9e-6, 1e-4, 1e-6), _row("Aa10", 30.0, 1e-6, 1e-5, 1e-7),
            _row("Bb10", 25.0, 1e-6, 5e-5, 3e-5)]
    for metric, winner in (("rate", "Aa10"), ("eoi", "Aa10"), ("eoc", "Bb10")):
        args = opt.parse_args(["Tc99_m1", "-p", "p", "--de", "1", "--metric", metric])
        text = opt.write_outputs(tmp_path / metric, "Tc99_m1", rows, [], args)
        optimal = next(ln for ln in text.splitlines() if ln.startswith("OPTIMAL"))
        assert f"OPTIMAL ({metric}): {winner}" in optimal
        assert "rate: Aa10(p,x)Tc99_m1 at Ebeam = 20 MeV" in text
        assert "eoc: Bb10(p,x)Tc99_m1 at Ebeam = 25 MeV" in text
    header = (tmp_path / "eoc" / "scan.csv").read_text().splitlines()[0]
    assert "m_EOC_g" in header and "m_rate" in header and "Bq" not in header


def test_molar_mass_from_endf_record(tmp_path):
    """AWR is read from the MF1/MT451 first line of the decay file; a nuclide
    without a record stops with LookupError instead of a guessed mass."""
    d = tmp_path / "files" / "decay"
    d.mkdir(parents=True)
    head = " 4.210000+4 9.900000+1         -1          0          2          11000 1451    1\n"
    (d / "z042").write_text(head)
    assert opt.molar_mass(str(tmp_path), 42, 100) == pytest.approx(99.0 * opt.NEUTRON_MASS_U)
    with pytest.raises(LookupError, match="Mo101"):
        opt.molar_mass(str(tmp_path), 42, 101)


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


def _paths():
    st = opt.State()
    opt.machine(st)
    return st


def _proton_lines(sym, A, ebeam, eback):
    return ["projectile p", f"element {sym}", f"mass {A}", "Ibeam 1", "Tirrad 1 d", "Tcool 1 d",
            f"Ebeam {ebeam}", f"Eback {eback:.6f}"]


@needs_lib
def test_mo100_route_mass_and_precursor_exclusion(tmp_path):
    """p + Mo-100 at 20 MeV makes Tc-99m. Its mass must equal (activity / lambda)
    * M / N_A from the same run; listing Mo99 (a beta- precursor) must exclude the
    route; ISOTOPIA's own Leff must match the requested thickness."""
    paths = _paths()
    t = opt.Target(42, 100, True)
    eback, _ = opt.EbackSolver(2, 42, 100).eback(20.0, 0.05)
    lines = _proton_lines("Mo", 100, 20.0, eback)
    di = (43, 99, 1)
    M = opt.molar_mass(paths.path, 43, 99)
    assert abs(M - 99) / 99 < 0.01                    # sanity: close to the mass number
    row = opt.run_one(opt.RunSpec(di, 2, t, 20.0, eback, lines, molar_mass=M))
    assert row["status"] == "ok" and row["m_EOC_g"] > 0
    assert row["Leff_cm"] == pytest.approx(0.05, rel=1e-3)
    st = opt.run(lines, out=io.StringIO(), outdir=str(tmp_path))
    key = (43, 99, 1)
    N_eoc = float(st.activity[key][opt.NUMTIME]) / float(st.lam[key])
    assert row["m_EOC_g"] == pytest.approx(N_eoc * M / float(opt.AVOGADRO), rel=1e-5)
    row = opt.run_one(opt.RunSpec(di, 2, t, 20.0, eback, lines, {(42, 99, None)}, M))
    assert row["status"] == "excluded" and "Mo99" in row["note"]


@needs_lib
def test_stable_di_mass(tmp_path):
    """Stable Ru-100 from p + Ru-101 at 30 MeV: previously rejected (no activity),
    now reported as a mass. A stable product cannot decay, so its mass cannot
    drop during cooling. Analytic check of the rate: at early times N = N_0 R t
    for direct production, so the first-step rate is N_0 R M / N_A per mA h.
    Tolerance 5%: beta+ feeding from Rh-100 adds a small second-order term."""
    paths = _paths()
    t = opt.Target(44, 101, True)
    eback, _ = opt.EbackSolver(2, 44, 101).eback(30.0, 0.05)
    lines = _proton_lines("Ru", 101, 30.0, eback)
    M = opt.molar_mass(paths.path, 44, 100)
    row = opt.run_one(opt.RunSpec((44, 100, 0), 2, t, 30.0, eback, lines, molar_mass=M))
    assert row["status"] == "ok" and row["m_EOI_g"] > 0
    assert row["m_EOC_g"] >= row["m_EOI_g"] * (1 - 1e-6)
    st = opt.run(lines, out=io.StringIO(), outdir=str(tmp_path))
    key = opt.di_key(st, (44, 100, 0))
    direct = float(st.N_0) * float(st.reaction_rate[key]) * M / float(opt.AVOGADRO)   # g/s
    assert row["m_rate"] == pytest.approx(direct * 3600 / (float(st.Ibeam) * 1e3), rel=0.05)
