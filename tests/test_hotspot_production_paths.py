"""Verify production callers, acceptance filtering, and CSV stopping semantics."""
import ast
import importlib.util
import os
from pathlib import Path

import pytest

ROOT=Path(os.environ.get('BINDCRAFT_SOURCE',Path(__file__).resolve().parents[1]))


def generic():
    pytest.importorskip('jax')
    spec=importlib.util.spec_from_file_location('generic_utils',ROOT/'functions/generic_utils.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_both_scoring_paths_supply_original_mapping_inputs():
    tree=ast.parse((ROOT/'bindcraft.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='score_interface']
    assert len(calls)==2
    assert {ast.unparse(c.args[0]) for c in calls}=={'trajectory_relaxed','mpnn_design_relaxed'}
    for call in calls:
        assert ast.unparse(call.args[5])=="target_settings['chains']"
        assert ast.unparse(call.args[6])=="target_settings['starting_pdb']"


def test_final_evaluated_row_is_written_before_stop():
    tree=ast.parse((ROOT/'bindcraft.py').read_text())
    # Inspect the actual production block without executing its design loop.
    blocks=[n.body for n in ast.walk(tree) if hasattr(n,'body') and isinstance(n.body,list)]
    for block in blocks:
        for i,node in enumerate(block[:-2]):
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='insert_data' and ast.unparse(node.value.args[0])=='trajectory_csv':
                assert 'check_hotspot_trajectory_failures(trajectory_csv, advanced_settings)' in ast.unparse(block[i+1])
                assert 'raise SystemExit(1)' in ast.unparse(block[i+2])
                return
    pytest.fail('Production stop block not found')


def test_csv_gate_includes_final_row_and_excludes_unevaluated(tmp_path):
    g=generic();p=str(tmp_path/'trajectory.csv')
    g.pd.DataFrame({'Target_HotspotContactPass':[False]*3+[True]*6+[None]*3}).to_csv(p,index=False)
    assert g.check_hotspot_trajectory_failures(p,{}) is False
    g.insert_data(p,[True])
    assert len(g.read_dataframe(p))==13
    assert g.check_hotspot_trajectory_failures(p,{}) is True
    g.pd.DataFrame({'Target_HotspotContactPass':[False]+[True]*9+[None]*3}).to_csv(p,index=False)
    assert g.check_hotspot_trajectory_failures(p,{}) is False
    assert g.check_hotspot_trajectory_failures(p,{'hotspot_contact_max_failure_fraction':0.1}) is True


def test_candidate_contact_metric_is_filtered_without_threshold_changes():
    g=generic();label='Average_Target_HotspotContactPass'
    filters={label:{'threshold':1,'higher':True}}
    assert g.check_filters([1.0],[label],filters) is True
    assert g.check_filters([0.0],[label],filters)==[label]


def test_saved_relaxed_trajectory_and_candidate_score_interface(tmp_path,monkeypatch):
    bank_path=os.environ.get('SAVED_PDL1_BANK')
    if not bank_path:
        pytest.skip('Saved structures are required for scientific-image integration')
    pytest.importorskip('pyrosetta')
    import importlib
    import sys
    import types
    import hashlib
    import pyrosetta as pr
    # Avoid eager package imports unrelated to this scorer; load real modules.
    package=types.ModuleType('production_functions')
    package.__path__=[str(ROOT/'functions')]
    monkeypatch.setitem(sys.modules,'production_functions',package)
    scorer=importlib.import_module('production_functions.pyrosetta_utils')
    b=importlib.import_module('production_functions.biopython_utils')
    pr.init('-mute all -ignore_unrecognized_res -ignore_zero_occupancy -corrections::beta_nov16 true -relax:default_repeats 1 -holes:dalphaball /app/bindcraft/functions/DAlphaBall.gcc')
    bank=Path(bank_path)
    structures=[bank/'campaign/Trajectory/Relaxed/PDL1_l80_s994205394.pdb',
                bank/'campaign/Accepted/PDL1_l80_s239321320_mpnn1_model1.pdb']
    # Accepted filename is recorded by the bank; do not assume model rank.
    structures[1]=next((bank/'campaign/Accepted').glob('PDL1_l80_s239321320_mpnn1*.pdb'))
    for pdb in structures:
        digest=hashlib.sha256(pdb.read_bytes()).hexdigest()
        scores,_,_=scorer.score_interface(str(pdb),'B','56',4.0,0.5,'A',str(ROOT/'example/PDL1.pdb'))
        assert scores['target_hotspot_contact_pass'] is True
        assert scores['target_hotspot_contacts']=='A39'
        # Rosetta export used by pr_relax retains existing PDB identifiers.
        roundtrip=tmp_path/pdb.name
        pr.pose_from_pdb(str(pdb)).dump_pdb(str(roundtrip))
        assert b.target_hotspot_contacts(str(roundtrip),'56',starting_pdb=str(ROOT/'example/PDL1.pdb'))['target_hotspot_residues']=='A39'
        assert hashlib.sha256(pdb.read_bytes()).hexdigest()==digest
