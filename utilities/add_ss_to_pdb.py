import re
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io import load_structure


def _set_cols(buf: List[str], start_col: int, end_col: int, text: str):
    """
    Place text into buf (list of characters) at 1-based columns start_col..end_col (inclusive).
    Truncates or pads text to fit.
    """
    assert start_col >= 1 and end_col >= start_col
    s = text[: (end_col - start_col + 1)].ljust(end_col - start_col + 1)
    buf[start_col - 1 : end_col] = list(s)


def _format_helix_record(
    ser_num: int,
    helix_id: str,
    init_resname: str,
    init_chain: str,
    init_seqnum: str,
    init_icode: str,
    end_resname: str,
    end_chain: str,
    end_seqnum: str,
    end_icode: str,
    helix_class: int = 1,
    length: int = 0,
) -> str:
    # Build an 80-char HELIX line according to PDB columns
    buf = list(" " * 80)
    _set_cols(buf, 1, 6, "HELIX")
    _set_cols(buf, 8, 10, f"{ser_num:>3}")
    _set_cols(buf, 12, 14, f"{helix_id:<3}")
    _set_cols(buf, 16, 18, f"{init_resname:>3}")
    _set_cols(buf, 20, 20, f"{init_chain:1}")
    _set_cols(buf, 22, 25, f"{int(init_seqnum):>4}" if init_seqnum not in ("", None) else "    ")
    _set_cols(buf, 26, 26, f"{init_icode or ' '}")
    _set_cols(buf, 28, 30, f"{end_resname:>3}")
    _set_cols(buf, 32, 32, f"{end_chain:1}")
    _set_cols(buf, 34, 37, f"{int(end_seqnum):>4}" if end_seqnum not in ("", None) else "    ")
    _set_cols(buf, 38, 38, f"{end_icode or ' '}")
    _set_cols(buf, 39, 40, f"{helix_class:>2}")
    _set_cols(buf, 72, 76, f"{length:>5}")
    return "".join(buf).rstrip() + "\n"


def _format_sheet_record(
    strand_num: int,
    sheet_id: str,
    init_resname: str,
    init_chain: str,
    init_seqnum: str,
    init_icode: str,
    end_resname: str,
    end_chain: str,
    end_seqnum: str,
    end_icode: str,
    sense: int = 0,
) -> str:
    # Build an 80-char SHEET line according to PDB columns (minimal fields populated)
    buf = list(" " * 80)
    _set_cols(buf, 1, 6, "SHEET")
    _set_cols(buf, 8, 10, f"{strand_num:>3}")
    _set_cols(buf, 12, 14, f"{sheet_id:<3}")
    _set_cols(buf, 16, 18, f"{init_resname:>3}")
    _set_cols(buf, 20, 20, f"{init_chain:1}")
    _set_cols(buf, 22, 25, f"{int(init_seqnum):>4}" if init_seqnum not in ("", None) else "    ")
    _set_cols(buf, 26, 26, f"{init_icode or ' '}")
    _set_cols(buf, 28, 30, f"{end_resname:>3}")
    _set_cols(buf, 32, 32, f"{end_chain:1}")
    _set_cols(buf, 33, 36, f"{int(end_seqnum):>4}" if end_seqnum not in ("", None) else "    ")
    _set_cols(buf, 37, 37, f"{end_icode or ' '}")
    _set_cols(buf, 38, 40, f"{sense:>3}")
    return "".join(buf).rstrip() + "\n"


def add_secondary_structure_to_pdb(
    input_pdb: str,
    output_pdb: str,
    min_helix_len: int = 4,
    min_sheet_len: int = 3,
    replace_existing: bool = True,
) -> str:
    """
    Read input_pdb, compute secondary structure by P-SEA (biotite.annotate_sse),
    insert HELIX and SHEET records into the header, and write to output_pdb.

    Parameters
    ----------
    input_pdb : str
        Path to source PDB file.
    output_pdb : str
        Path to write annotated PDB file (overwrites if exists).
    min_helix_len : int
        Minimum consecutive 'a' residues to produce a HELIX record.
    min_sheet_len : int
        Minimum consecutive 'b' residues to produce a SHEET record.
    replace_existing : bool
        If True, any existing HELIX/SHEET lines in the original PDB will be removed
        before inserting new ones.
    """
    # read original raw lines (preserve as much as possible)
    p = Path(input_pdb)
    raw_lines = p.read_text().splitlines(keepends=True)

    # load structure via Biotite
    pdb = PDBFile.read(str(input_pdb))
    assert pdb.get_model_count() == 1, "This routine expects single-model PDBs"
    atom_array = pdb.get_structure()[0]

    # compute sse (one label per residue)
    sse = struc.annotate_sse(atom_array)

    # Build unique residue list (first occurrence order)
    chain_arr = np.asarray(atom_array.chain_id, dtype="U")
    res_arr = np.asarray(atom_array.res_id, dtype="U")
    resname_arr = np.asarray(atom_array.res_name, dtype="U")

    seen = {}
    unique_chain = []
    unique_seq = []
    unique_resname = []
    for i, (c, r, rn) in enumerate(zip(chain_arr, res_arr, resname_arr)):
        key = f"{c}:{r}"
        if key not in seen:
            seen[key] = i
            unique_chain.append(c)
            unique_seq.append(r)
            unique_resname.append(rn)

    # sse length vs unique residues
    if len(sse) != len(unique_seq):
        warnings.warn(
            f"Length mismatch: annotate_sse returned {len(sse)} entries but "
            f"found {len(unique_seq)} unique residues. Mapping will use min length."
        )
    L = min(len(sse), len(unique_seq))
    unique_chain = unique_chain[:L]
    unique_seq = unique_seq[:L]
    unique_resname = unique_resname[:L]
    sse = sse[:L]

    # helper to split a residue id into seqnum and insertion code (if present)
    def _split_seq_and_icode(seq_str: str) -> Tuple[str, str]:
        # match optional leading sign and digits, optional trailing letter (insertion code)
        m = re.match(r"^(-?\d+)(\w?)$", str(seq_str))
        if m:
            return m.group(1), (m.group(2) if m.group(2) and m.group(2) != " " else "")
        else:
            # fallback: try to extract integer and remaining
            m2 = re.match(r"^(-?\d+)", str(seq_str))
            if m2:
                rest = str(seq_str)[ len(m2.group(1)) : ] or ""
                return m2.group(1), (rest if rest != " " else "")
            else:
                return seq_str, ""  # opaque fallback

    # find contiguous segments
    helix_segments = []
    sheet_segments = []

    i = 0
    while i < L:
        label = sse[i]
        if label == "a":
            j = i + 1
            while j < L and sse[j] == "a":
                j += 1
            seg_len = j - i
            if seg_len >= min_helix_len:
                helix_segments.append((i, j - 1))
            i = j
        elif label == "b":
            j = i + 1
            while j < L and sse[j] == "b":
                j += 1
            seg_len = j - i
            if seg_len >= min_sheet_len:
                sheet_segments.append((i, j - 1))
            i = j
        else:
            i += 1

    # create HELIX / SHEET strings
    helix_lines = []
    for idx, (start_idx, end_idx) in enumerate(helix_segments, start=1):
        init_resname = unique_resname[start_idx]
        init_chain = unique_chain[start_idx]
        init_seq, init_icode = _split_seq_and_icode(unique_seq[start_idx])
        end_resname = unique_resname[end_idx]
        end_chain = unique_chain[end_idx]
        end_seq, end_icode = _split_seq_and_icode(unique_seq[end_idx])
        length = int(end_seq) - int(init_seq) + 1 if init_seq.isdigit() and end_seq.isdigit() else (end_idx - start_idx + 1)
        helix_id = f"H{idx}"
        helix_lines.append(
            _format_helix_record(
                ser_num=idx,
                helix_id=helix_id,
                init_resname=init_resname,
                init_chain=init_chain,
                init_seqnum=init_seq,
                init_icode=init_icode,
                end_resname=end_resname,
                end_chain=end_chain,
                end_seqnum=end_seq,
                end_icode=end_icode,
                helix_class=1,
                length=length,
            )
        )

    sheet_lines = []
    for idx, (start_idx, end_idx) in enumerate(sheet_segments, start=1):
        init_resname = unique_resname[start_idx]
        init_chain = unique_chain[start_idx]
        init_seq, init_icode = _split_seq_and_icode(unique_seq[start_idx])
        end_resname = unique_resname[end_idx]
        end_chain = unique_chain[end_idx]
        end_seq, end_icode = _split_seq_and_icode(unique_seq[end_idx])
        sheet_id = f"S{idx}"
        sense = 0
        sheet_lines.append(
            _format_sheet_record(
                strand_num=idx,
                sheet_id=sheet_id,
                init_resname=init_resname,
                init_chain=init_chain,
                init_seqnum=init_seq,
                init_icode=init_icode,
                end_resname=end_resname,
                end_chain=end_chain,
                end_seqnum=end_seq,
                end_icode=end_icode,
                sense=sense,
            )
        )

    # prepare a new header: remove existing HELIX/SHEET lines if requested
    new_lines: List[str] = []
    inserted = False
    first_atom_idx = None
    for i, ln in enumerate(raw_lines):
        if ln.startswith(("ATOM  ", "HETATM")):
            first_atom_idx = i
            break

    if first_atom_idx is None:
        raise ValueError("No ATOM/HETATM records found in input PDB.")

    # take all header lines before first_atom_idx, filter HELIX/SHEET if replace_existing
    header_lines = raw_lines[:first_atom_idx]
    if replace_existing:
        header_lines = [ln for ln in header_lines if not (ln.startswith("HELIX") or ln.startswith("SHEET"))]

    # choose insertion point: after header_lines, before ATOM block
    new_lines.extend(header_lines)

    # optionally add a REMARK indicating generation
    new_lines.append(f"REMARK   1 GENERATED_BY: add_secondary_structure_to_pdb\n")

    # append HELIX and SHEET records (HELIX first per PDB conventions)
    new_lines.extend(helix_lines)
    new_lines.extend(sheet_lines)

    # then append the rest of the original file (ATOM/HETATM + everything after)
    new_lines.extend(raw_lines[first_atom_idx:])

    # write out
    out_path = Path(output_pdb)
    out_path.write_text("".join(new_lines))
    return str(out_path)

if __name__ == '__main__':
    annotated = add_secondary_structure_to_pdb("cd72.pdb", "cd72_with_ss.pdb")
    print("Wrote:", annotated)