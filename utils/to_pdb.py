import sys
from np import residue_constants as resc

#from nufold
def to_pdb(atom_coords, fas_np, b_factor=1.0, seq_type='dna') -> str:


    pdb_lines = []

    pdb_lines.append("MODEL     1")
    atom_index = 1
    chain_id = "A"

    for idx_res, i in enumerate(fas_np, start=1):
        res_name1 = resc.rev_restype1_order[i]
        res_name_3 = resc.restype_1to3[seq_type][res_name1]
        for atom_name in resc.residue_atoms[res_name_3]:
            atom_posi = atom_coords[idx_res-1][resc.atom_order[atom_name]]

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_id:>1}"
                f"{idx_res:>4}{insertion_code:>1}   "
                f"{atom_posi[0]:>8.3f}{atom_posi[1]:>8.3f}{atom_posi[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the chain.
    chain_end = "TER"
    chain_termination_line = (
        f"{chain_end:<6}{atom_index:>5}      {res_name_3:>3} "
        f"{chain_id:>1}{idx_res:>4}"
    )
    pdb_lines.append(chain_termination_line)
    pdb_lines.append("ENDMDL")

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)