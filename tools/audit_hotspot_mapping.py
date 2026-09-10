#!/usr/bin/env python3
"""Recheck saved structures without modifying the bank or running prediction."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import importlib.util
from Bio.PDB import PDBParser
spec = importlib.util.spec_from_file_location('biopython_utils', Path(os.environ.get('BINDCRAFT_SOURCE', Path(__file__).resolve().parents[1]))/'functions/biopython_utils.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
parse_target_hotspot_residues = b.parse_target_hotspot_residues
target_hotspot_contacts = b.target_hotspot_contacts

def cached_structure(path):
    return PDBParser(QUIET=True).get_structure('audit', path)


def nearest(path, residues):
    model=cached_structure(str(path))[0]
    binder=np.array([a.coord for a in model['B'].get_atoms() if a.element not in ('H','D')])
    atoms=np.array([a.coord for chain,n in residues for a in model[chain][n] if a.element not in ('H','D')])
    return float(np.linalg.norm(atoms[:,None]-binder[None,:],axis=-1).min())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--starting-pdb',type=Path,required=True)
    parser.add_argument('--previous-output-hotspot',required=True,help='Previously assessed output residue numbers, for distance comparison only')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();bank=args.bank.resolve();out=args.output.resolve()
    if out==bank or bank in out.parents:raise ValueError('Audit output must be outside the saved bank')
    out.mkdir(parents=True,exist_ok=False)
    load=lambda p:json.loads(p.read_text())
    settings=load(bank/'settings.json');advanced=load(bank/'advanced.json')
    hotspot=settings['target_hotspot_residues'];chains=settings['chains']
    cutoff=advanced['hotspot_contact_distance_cutoff'];fraction=advanced['hotspot_contact_required_fraction']
    corrected=parse_target_hotspot_residues(hotspot,chains,starting_pdb=str(args.starting_pdb))
    previous=parse_target_hotspot_residues(args.previous_output_hotspot)
    root=bank/'campaign';meta=root/'.bindcraftplus'
    manifest=load(meta/'manifest.json');head=load(meta/'cursor.json')['next_index']
    outcomes=[load(meta/'outcomes'/f"{j['id']}.json") for j in manifest['jobs'][:head]]
    rows=[];accepted=[]
    for r in outcomes:
        job=r['job']
        for row in r['tables']['trajectory_stats.csv']:
            pdb=root/'Trajectory/Relaxed'/f"{row['Design']}.pdb"
            result=target_hotspot_contacts(str(pdb),hotspot,target_chains=chains,starting_pdb=str(args.starting_pdb),atom_distance_cutoff=cutoff,required_fraction=fraction)
            old_metrics=target_hotspot_contacts(str(pdb),args.previous_output_hotspot,atom_distance_cutoff=cutoff,required_fraction=fraction)
            assert old_metrics['target_hotspot_contact_pass'] == row['Target_HotspotContactPass']
            rows.append({'root':job['id'],'root_number':job['index']+1,'design':row['Design'],
                         'old_pass':row['Target_HotspotContactPass'],'corrected_pass':result['target_hotspot_contact_pass'],
                         'old_distance_A':nearest(pdb,previous),'corrected_distance_A':nearest(pdb,corrected),
                         'structure_sha256':hashlib.sha256(pdb.read_bytes()).hexdigest()})
        original_bank=Path(settings['design_path']).parent
        for name,entry in r.get('accepted_structures',{}).items():
            pdb=bank/Path(entry['path']).relative_to(original_bank)
            assert hashlib.sha256(pdb.read_bytes()).hexdigest()==entry['sha256']
            metrics=target_hotspot_contacts(str(pdb),hotspot,target_chains=chains,starting_pdb=str(args.starting_pdb),atom_distance_cutoff=cutoff,required_fraction=fraction)
            accepted.append({'name':name,'sha256':entry['sha256'],'corrected_hotspot_pass':metrics['target_hotspot_contact_pass']})
    report={'analysis':'CORRECTED OFFLINE ANALYSIS WITH PRODUCTION BINDCRAFT; no pipeline run', 'source_pdb_sha256':hashlib.sha256(args.starting_pdb.read_bytes()).hexdigest(), 'mapper_sha256':hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(), 'source_hotspot':hotspot,'corrected_output_hotspots':sorted(corrected),'previous_output_hotspots':sorted(previous),
            'cutoff_A':cutoff,'required_fraction':fraction,'evaluated':len(rows),'old_failures':sum(r['old_pass'] is False for r in rows),
            'corrected_failures':sum(r['corrected_pass'] is False for r in rows),'roots_committed':head,'rows':rows,'historically_accepted':accepted,
            'limitation':'Only saved structures are rechecked. Missing assessment suffixes and complete acceptance reclassification are not evaluated; the original bank is unchanged.'}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'contacts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/bcp-hotspot-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
    x=np.arange(len(rows));width=.36
    fmt=lambda residues:','.join(f'{c}{n}' for c,n in sorted(residues))
    ax.bar(x-width/2,[r['old_distance_A'] for r in rows],width,color='#a6bbc8',label=f'Previously checked output {fmt(previous)}')
    ax.bar(x+width/2,[r['corrected_distance_A'] for r in rows],width,color='#179c7d',label=f'Correct output {fmt(corrected)} (source {hotspot})')
    ax.axhline(cutoff,color='#f58220',ls='--',lw=1.5,label=f'{cutoff:g} Å contact cutoff')
    ax.set(xticks=x,xticklabels=[r['root_number'] for r in rows],xlabel='Root number (only evaluated relaxed structures)',ylabel='Nearest binder heavy atom (Å)',
           title=f"Hotspot mapping: {report['old_failures']}/{len(rows)} recorded failures → {report['corrected_failures']}/{len(rows)} corrected")
    ax.legend(frameon=False,loc='upper left');ax.set_ylim(0,max(r['old_distance_A'] for r in rows)+2.5)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'hotspot-mapping.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
