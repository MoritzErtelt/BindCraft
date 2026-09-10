####################################
################ BioPython functions
####################################
### Import dependencies
import os
import io
import math
import re
import numpy as np
from functools import lru_cache
from collections import defaultdict
from scipy.spatial import cKDTree
from Bio.PDB.PDBExceptions import PDBConstructionException
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

def _normalise_target_chain_ids(target_chains):
    return [chain_id.strip() for chain_id in str(target_chains).split(',') if chain_id.strip()]

def _parse_target_hotspot_tokens(target_hotspot_residues):
    hotspot_string = str(target_hotspot_residues).strip()
    if not hotspot_string or hotspot_string.lower() in ["none", "null", "false"]:
        return None

    hotspot_tokens = []
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

        start_chain = match.group(1) if match.group(1) else None
        end_chain = match.group(3) if match.group(3) else start_chain
        if start_chain and end_chain and start_chain != end_chain:
            raise ValueError(f"Invalid cross-chain hotspot range: {token}")

        start = int(match.group(2))
        end = int(match.group(4) or start)
        if end < start:
            raise ValueError(f"Invalid target_hotspot_residues range: {token}")

        hotspot_tokens.append((token, start_chain, start, end))

    return hotspot_tokens or None

def _strict_hotspot_structure(data):
    """Do not let BioPython silently discard duplicate residue/atom identifiers."""
    try:
        return PDBParser(QUIET=True, PERMISSIVE=False).get_structure("hotspot", io.StringIO(data))
    except PDBConstructionException as exc:
        raise ValueError(f"Ambiguous PDB identifiers in hotspot mapping: {exc}") from exc


def _build_target_hotspot_residue_map(starting_pdb, target_chains, trajectory_target_chain="A"):
    # Content identity prevents stale correspondence if a source path is reused.
    with open(starting_pdb) as handle:
        data = handle.read()
    lookup, numbers = _target_hotspot_map_by_content(data, target_chains, trajectory_target_chain)
    return dict(lookup), dict(numbers)


@lru_cache(maxsize=32)
def _target_hotspot_map_by_content(data, target_chains, trajectory_target_chain):
    structure = _strict_hotspot_structure(data)
    model = structure[0]

    chain_ids = _normalise_target_chain_ids(target_chains)
    if not chain_ids:
        raise ValueError("No target chains provided for hotspot mapping")

    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError("Duplicate target chains make hotspot mapping ambiguous")

    residue_lookup = {}
    residue_number_lookup = defaultdict(list)
    previous_last_residue_index = None
    first_residue_index = None
    exported_identifiers = set()

    for chain_id in chain_ids:
        if chain_id not in model:
            raise ValueError(f"Target chain {chain_id} not found in source PDB")

        for residue in model[chain_id]:
            if residue.is_disordered() == 2:
                raise ValueError("Ambiguous alternate target residue identities")
            if residue.id[2].strip():
                raise ValueError("Insertion-coded target residues require unambiguous renumbering before design")
            if 'N' in residue and (not is_aa(residue, standard=True) or residue.id[0] != ' '):
                raise ValueError("Nonstandard target residues require explicit normalization before hotspot mapping")

        residues = [residue for residue in model[chain_id] if is_aa(residue, standard=True) and 'N' in residue]
        if not residues:
            raise ValueError(f"Target chain {chain_id} has no standard amino-acid residues with backbone N atoms in source PDB")

        residue_offset = 0 if previous_last_residue_index is None else previous_last_residue_index + 50
        chain_residue_indices = []

        for residue in residues:
            residue_id = residue.id[1]
            residue_index = residue_id + residue_offset
            residue_key = (chain_id, residue_id)
            if first_residue_index is None:
                first_residue_index = residue_index
            # prep_pdb retains numbering gaps and adds the previous index + 50
            # for each source chain. Binder save_pdb merges the target into A,
            # then renum_pdb_str starts that merged chain at 1.
            residue_value = (trajectory_target_chain, residue_index - first_residue_index + 1)
            if residue_key in residue_lookup or residue_value in exported_identifiers:
                raise ValueError("Ambiguous source-to-output hotspot residue identifiers")
            exported_identifiers.add(residue_value)
            residue_lookup[residue_key] = residue_value
            residue_number_lookup[residue_id].append(residue_value)
            chain_residue_indices.append(residue_index)

        previous_last_residue_index = chain_residue_indices[-1]

    return residue_lookup, {residue_id: tuple(values) for residue_id, values in residue_number_lookup.items()}

def parse_target_hotspot_residues(target_hotspot_residues, target_chains="A", trajectory_target_chain="A", starting_pdb=None):
    if target_hotspot_residues in [None, False]:
        return None

    hotspot_tokens = _parse_target_hotspot_tokens(target_hotspot_residues)
    if hotspot_tokens is None:
        return None

    target_chain_ids = _normalise_target_chain_ids(target_chains)
    if starting_pdb is None:
        if len(target_chain_ids) != 1:
            raise ValueError("Explicit multichain hotspots require the source PDB for mapping")

        source_target_chain = target_chain_ids[0]
        hotspot_residues = set()
        for token, start_chain, start, end in hotspot_tokens:
            if start_chain and start_chain != source_target_chain:
                raise ValueError(f"Hotspot {token} does not match target chain {source_target_chain}")

            for residue_id in range(start, end + 1):
                hotspot_residues.add((trajectory_target_chain, residue_id))

        return hotspot_residues or None

    residue_lookup, residue_number_lookup = _build_target_hotspot_residue_map(starting_pdb, target_chains, trajectory_target_chain)
    hotspot_residues = set()
    missing_hotspots = []
    default_target_chain = target_chain_ids[0]

    for token, start_chain, start, end in hotspot_tokens:
        for residue_id in range(start, end + 1):
            hotspot_chain = start_chain or default_target_chain
            mapped_residue = residue_lookup.get((hotspot_chain, residue_id))
            if mapped_residue is None:
                missing_hotspots.append(f"{hotspot_chain}{residue_id}" if start_chain else str(residue_id))
                continue
            hotspot_residues.add(mapped_residue)

    if missing_hotspots:
        missing_hotspots = ','.join(sorted(set(missing_hotspots), key=lambda value: (re.sub(r'\d+', '', value), int(re.search(r'\d+', value).group()) if re.search(r'\d+', value) else -1)))
        raise ValueError(f"Could not map target hotspot residues from {starting_pdb}: {missing_hotspots}")

    return hotspot_residues or None

def _format_target_hotspot_residues(hotspot_residues):
    return ','.join(f"{chain}{residue_id}" for chain, residue_id in sorted(hotspot_residues, key=lambda item: (item[0], item[1])))

def target_hotspot_contacts(pdb_file, target_hotspot_residues, binder_chain="B", target_chain="A", target_chains="A", atom_distance_cutoff=4.0, required_fraction=0.5, starting_pdb=None):
    parsed_hotspots = parse_target_hotspot_residues(target_hotspot_residues, target_chains, target_chain, starting_pdb)
    if parsed_hotspots is None:
        return {
            'target_hotspot_residues': None,
            'target_hotspot_contacts': None,
            'target_hotspot_contact_count': None,
            'target_hotspot_contact_fraction': None,
            'target_hotspot_contact_pass': None,
        }

    with open(pdb_file) as handle:
        structure = _strict_hotspot_structure(handle.read())
    model = structure[0]

    if binder_chain not in model:
        raise ValueError(f"Binder chain {binder_chain} not found in {pdb_file}")
    if target_chain not in model:
        raise ValueError(f"Target chain {target_chain} not found in {pdb_file}")

    mapped = set()
    for residue in model[target_chain]:
        key = (target_chain, residue.id[1])
        if key in parsed_hotspots:
            if residue.is_disordered() == 2 or residue.id[2].strip() or residue.id[0] != ' ' or key in mapped:
                raise ValueError(f"Ambiguous assessed hotspot identifier: {key}")
            mapped.add(key)
    if parsed_hotspots - mapped:
        raise ValueError(f"Mapped explicit hotspots missing from assessed structure: {sorted(parsed_hotspots - mapped)}")

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
