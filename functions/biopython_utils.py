####################################
################ BioPython functions
####################################
### Import dependencies
import os
import math
import re
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, Superimposer
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa

# analyze sequence composition of design
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []

    # Check if protein contains clashes after relaxation
    if num_clashes > 0:
        note_array.append('Relaxed structure contains clashes.')

    # Check if the sequence contains disallowed amino acids
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append('Contains: '+restricted_AA+'!')

    # Analyze the protein
    analysis = ProteinAnalysis(sequence)

    # Calculate the reduced extinction coefficient per 1% solution
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # Check if the absorption is high enough
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(f'Absorption value is {extinction_coefficient_reduced_1}, consider adding tryptophane to design.')

    # Join the notes into a single string
    notes = ' '.join(note_array)

    return notes

# temporary function, calculate RMSD of input PDB and trajectory target
def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)
    
    # Extract chain A from trajectory_pdb
    chain_trajectory = structure_trajectory[0]['A']
    
    # Extract the specified chains from starting_pdb
    chain_ids = chain_ids_string.split(',')
    residues_starting = []
    for chain_id in chain_ids:
        chain_id = chain_id.strip()
        chain = structure_starting[0][chain_id]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)
    
    # Extract residues from chain A in trajectory_pdb
    residues_trajectory = [residue for residue in chain_trajectory if is_aa(residue, standard=True)]
    
    # Ensure that both structures have the same number of residues
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]
    
    # Collect CA atoms from the two sets of residues
    atoms_starting = [residue['CA'] for residue in residues_starting if 'CA' in residue]
    atoms_trajectory = [residue['CA'] for residue in residues_trajectory if 'CA' in residue]
    
    # Calculate RMSD using structural alignment
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms
    
    return round(rmsd, 2)

# detect C alpha clashes for deformed trajectories
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []  # Detailed atom info for debugging and processing

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)

three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def parse_target_hotspot_residues(target_hotspot_residues, target_chain="A"):
    if target_hotspot_residues in [None, False]:
        return None

    hotspot_string = str(target_hotspot_residues).strip()
    if not hotspot_string or hotspot_string.lower() in ["none", "null", "false"]:
        return None

    hotspot_residues = set()
    for token in hotspot_string.split(','):
        token = token.strip()
        if not token:
            continue

        # Whole-chain hotspot settings are valid for design loss but too broad for this hard check.
        if re.fullmatch(r"[A-Za-z]+", token):
            return None

        match = re.fullmatch(r"([A-Za-z])?(\d+)(?:-([A-Za-z])?(\d+))?", token)
        if match is None:
            raise ValueError(f"Invalid target_hotspot_residues token: {token}")

        start = int(match.group(2))
        end = int(match.group(4) or start)
        if end < start:
            raise ValueError(f"Invalid target_hotspot_residues range: {token}")

        for residue_id in range(start, end + 1):
            hotspot_residues.add((target_chain, residue_id))

    return hotspot_residues or None

def _format_target_hotspot_residues(hotspot_residues):
    return ','.join(f"{chain}{residue_id}" for chain, residue_id in sorted(hotspot_residues, key=lambda item: (item[0], item[1])))

def target_hotspot_contacts(pdb_file, target_hotspot_residues, binder_chain="B", target_chain="A", atom_distance_cutoff=4.0, required_fraction=0.5):
    parsed_hotspots = parse_target_hotspot_residues(target_hotspot_residues, target_chain)
    if parsed_hotspots is None:
        return {
            'target_hotspot_residues': None,
            'target_hotspot_contacts': None,
            'target_hotspot_contact_count': None,
            'target_hotspot_contact_fraction': None,
            'target_hotspot_contact_pass': None,
        }

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)
    model = structure[0]

    if binder_chain not in model:
        raise ValueError(f"Binder chain {binder_chain} not found in {pdb_file}")
    if target_chain not in model:
        raise ValueError(f"Target chain {target_chain} not found in {pdb_file}")

    binder_atoms = [
        atom for atom in Selection.unfold_entities(model[binder_chain], 'A')
        if atom.element != 'H'
    ]
    target_atoms = []
    target_atom_residues = []

    for atom in Selection.unfold_entities(model[target_chain], 'A'):
        if atom.element == 'H':
            continue

        residue = atom.get_parent()
        residue_key = (target_chain, residue.id[1])
        if residue_key in parsed_hotspots:
            target_atoms.append(atom)
            target_atom_residues.append(residue_key)

    if not binder_atoms or not target_atoms:
        contacted_hotspots = set()
    else:
        binder_coords = np.array([atom.coord for atom in binder_atoms])
        target_coords = np.array([atom.coord for atom in target_atoms])
        binder_tree = cKDTree(binder_coords)
        target_tree = cKDTree(target_coords)
        pairs = target_tree.query_ball_tree(binder_tree, atom_distance_cutoff)
        contacted_hotspots = {
            target_atom_residues[target_idx]
            for target_idx, close_indices in enumerate(pairs)
            if close_indices
        }

    hotspot_count = len(parsed_hotspots)
    contact_count = len(contacted_hotspots)
    required_contacts = math.ceil(hotspot_count * required_fraction)

    return {
        'target_hotspot_residues': _format_target_hotspot_residues(parsed_hotspots),
        'target_hotspot_contacts': _format_target_hotspot_residues(contacted_hotspots),
        'target_hotspot_contact_count': contact_count,
        'target_hotspot_contact_fraction': round(contact_count / hotspot_count, 2),
        'target_hotspot_contact_pass': contact_count >= required_contacts,
    }

# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

# calculate secondary structure percentage of design
def calc_ss_percentage(pdb_file, advanced_settings, chain_id="B", atom_distance_cutoff=4.0):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100,2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100,2) if total > 0 else 0
    loop_percentage = round(((total - helix - sheet) / total) * 100,2) if total > 0 else 0

    return helix_percentage, sheet_percentage, loop_percentage
