"""Regression checks for the production source-to-export residue correspondence."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from Bio.PDB import PDBParser

# Load the scientific utility directly; package __init__ eagerly imports AF2/PyRosetta.
ROOT = Path(os.environ.get('BINDCRAFT_SOURCE', Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('biopython_utils', ROOT/'functions/biopython_utils.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def atom(serial, chain, residue, x, name='N', element='N'):
    return f'ATOM  {serial:5d} {name:^4s} ALA {chain}{residue:4d}    {x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{80.:6.2f}          {element:>2s}\n'


def write_pdb(path, residues):
    path.write_text(''.join(atom(i+1,c,n,x) for i,(c,n,x) in enumerate(residues))+'END\n')
    return str(path)


@pytest.mark.parametrize('intended,unrelated,expected', [(10,30,True),(30,10,False)])
def test_offset_source_false_negative_and_false_positive(tmp_path, intended, unrelated, expected):
    source=write_pdb(tmp_path/'source.pdb',[('A',18,0),('A',56,1),('A',73,2)])
    assessed=write_pdb(tmp_path/'assessed.pdb',[('A',1,0),('A',39,intended),('A',56,unrelated),('B',1,13)])
    result=b.target_hotspot_contacts(assessed,'56',starting_pdb=source)
    assert result['target_hotspot_residues']=='A39'
    assert result['target_hotspot_contact_pass'] is expected


def test_case_sensitive_chain_and_output(tmp_path):
    source=write_pdb(tmp_path/'case.pdb',[('A',18,0),('a',56,10)])
    assert b.parse_target_hotspot_residues('a56','a',starting_pdb=source)=={('A',1)}
    assert b.parse_target_hotspot_residues('a56','A,a',trajectory_target_chain='z',starting_pdb=source)=={('z',107)}
    with pytest.raises(ValueError,match='Could not map'):
        b.parse_target_hotspot_residues('A56','a',starting_pdb=source)


def test_gaps_and_missing_backbone_n(tmp_path):
    source=tmp_path/'missing.pdb'
    source.write_text(atom(1,'A',18,0,'CA','C')+atom(2,'A',19,1)+atom(3,'A',21,2,'CA','C')+atom(4,'A',22,3)+'END\n')
    assert b.parse_target_hotspot_residues('22',starting_pdb=str(source))=={('A',4)}
    for n in ('18','21'):
        with pytest.raises(ValueError,match='Could not map'):
            b.parse_target_hotspot_residues(n,starting_pdb=str(source))


@pytest.mark.parametrize('kind', ['insertion','duplicate_chain','duplicate_residue','duplicate_atom'])
def test_ambiguous_source_is_rejected(tmp_path,kind):
    p=tmp_path/'ambiguous.pdb'; line=atom(1,'A',18,0)
    if kind=='insertion': data=line[:26]+'A'+line[27:]
    elif kind=='duplicate_residue': data=line+atom(2,'A',19,1)+atom(3,'A',18,2)
    elif kind=='duplicate_atom': data=line+atom(2,'A',18,2)
    else: data=line
    p.write_text(data+'END\n')
    with pytest.raises(ValueError,match='[Aa]mbiguous|Insertion-coded|Duplicate'):
        b.parse_target_hotspot_residues('18','A,A' if kind=='duplicate_chain' else 'A',starting_pdb=str(p))


def test_same_filename_new_content_changes_mapping(tmp_path):
    p=tmp_path/'source.pdb'
    source=write_pdb(p,[('A',18,0),('A',56,1)])
    assert b.parse_target_hotspot_residues('56',starting_pdb=source)=={('A',39)}
    write_pdb(p,[('A',19,0),('A',56,1)])
    assert b.parse_target_hotspot_residues('56',starting_pdb=source)=={('A',38)}


@pytest.mark.parametrize('malformed', ['missing','insertion','duplicate'])
def test_assessed_hotspot_must_be_unambiguous_and_present(tmp_path,malformed):
    source=write_pdb(tmp_path/'source.pdb',[('A',18,0),('A',56,1)])
    line=atom(2,'A',39,10)
    if malformed=='missing': line=''
    elif malformed=='insertion': line=line[:26]+'A'+line[27:]
    else: line+=atom(3,'A',39,11)
    p=tmp_path/'output.pdb';p.write_text(atom(1,'A',1,0)+line+atom(4,'B',1,13)+'END\n')
    with pytest.raises(ValueError): b.target_hotspot_contacts(str(p),'56',starting_pdb=source)


@pytest.mark.parametrize('distance,passed',[(4.0,True),(4.001,False)])
def test_distance_and_fraction_boundaries_unchanged(tmp_path,distance,passed):
    source=write_pdb(tmp_path/'source.pdb',[('A',18,0),('A',19,20),('A',20,40)])
    pdb=write_pdb(tmp_path/'output.pdb',[('A',1,0),('A',2,20),('A',3,40),('B',1,distance)])
    result=b.target_hotspot_contacts(pdb,'18-19',starting_pdb=source)
    assert result['target_hotspot_contact_pass'] is passed
    assert b.target_hotspot_contacts(pdb,'18-20',starting_pdb=source)['target_hotspot_contact_pass'] is False


@pytest.mark.parametrize('chains',['X,Y','Y,X','a,A'])
@pytest.mark.parametrize('missing_first_n',[False,True])
def test_native_colabdesign_preparation_and_actual_save_pdb(tmp_path,chains,missing_first_n):
    # Required in both production image validation lanes; skips only on a CPU host
    # without the production dependencies. No model construction or inference.
    pytest.importorskip('colabdesign')
    from colabdesign.af.prep import prep_pdb
    from colabdesign.shared.prep import prep_pos
    from colabdesign.af.utils import _af_utils
    first,second=chains.split(',')
    c1,c2=('X','Y') if 'X' in chains else ('A','a')
    residues=[(c1,18),(c1,19),(c1,22),(c2,5),(c2,7)]
    source=tmp_path/'native.pdb'
    source.write_text(''.join(atom(i+1,c,n,float(i), 'CA' if i==0 and missing_first_n else 'N', 'C' if i==0 and missing_first_n else 'N') for i,(c,n) in enumerate(residues))+'END\n')
    prepared=prep_pdb(str(source),chain=chains,ignore_missing=True)
    idx=prepared['residue_index']; batch=prepared['batch']; size=len(idx)
    aux={'all':{'aatype':batch['aatype'][None], 'residue_index':idx[None],
        'atom_positions':batch['all_atom_positions'][None], 'atom_mask':batch['all_atom_mask'][None],
        'plddt':np.ones((1,size))}}
    output=tmp_path/'exported.pdb'
    _af_utils.save_pdb(SimpleNamespace(_lengths=[size]),str(output),aux=aux)
    saved=list(PDBParser(QUIET=True).get_structure('saved',str(output))[0]['A'])
    for c,n in residues[1 if missing_first_n else 0:]:
        position=int(prep_pos(f'{c}{n}',**prepared['idx'])['pos'][0])
        assert b.parse_target_hotspot_residues(f'{c}{n}',chains,starting_pdb=str(source))=={('A',saved[position].id[1])}
    # Candidate initial-guess preparation starts from the exported target.
    reloaded=prep_pdb(str(output),chain='A',ignore_missing=True)
    assert np.array_equal(reloaded['residue_index']-reloaded['residue_index'][0]+1,[r.id[1] for r in saved])
    # Unqualified numbers use the first requested source chain, as prep_pos does.
    number=22 if first==c1 else 7
    assert b.parse_target_hotspot_residues(str(number),chains,starting_pdb=str(source))==b.parse_target_hotspot_residues(f'{first}{number}',chains,starting_pdb=str(source))


def test_disordered_residue_identity_is_rejected(tmp_path):
    # Two amino-acid identities at one residue number are not a unique mapping.
    first=atom(1,'A',18,0);second=atom(2,'A',18,1)
    first=first[:16]+'A'+first[17:]
    second=second[:16]+'BGLY'+second[20:]
    p=tmp_path/'disordered.pdb';p.write_text(first+second+'END\n')
    with pytest.raises(ValueError,match='[Aa]mbiguous'):
        b.parse_target_hotspot_residues('18',starting_pdb=str(p))


def test_nonstandard_source_fails_closed(tmp_path):
    p=tmp_path/'modified.pdb';p.write_text(atom(1,'A',18,0).replace('ALA','MSE')+'END\n')
    with pytest.raises(ValueError,match='Nonstandard'):
        b.parse_target_hotspot_residues('18',starting_pdb=str(p))
