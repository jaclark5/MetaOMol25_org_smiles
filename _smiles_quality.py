"""
Per-molecule SMILES quality checks.

Imported by 2_get_smiles_sharded.py to add quality columns to each metadata
record, and by 5_smiles_quality.py to classify SMILES already on the Hub.

The SMILES are atom-mapped with explicit hydrogens, so they are parsed with
removeHs=False and sanitize=False; sanitization is only attempted on a copy.
"""

from typing import Any, Dict

from rdkit import Chem


def parse_mapped_smiles(smiles: str):
    """Parse a SMILES without removing hydrogens or sanitizing. Returns None on failure."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    return Chem.MolFromSmiles(smiles, params)


def count_orphaned_h(mol) -> int:
    """Number of hydrogen atoms with no bonds (e.g. a detached [H+])."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0)


def try_sanitize(mol):
    """Sanitize a copy of mol. Returns (sanitized copy or None, exception name or "")."""
    test_mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(test_mol)
    except Exception as e:
        return None, type(e).__name__
    return test_mol, ""


def count_radical_electrons(sanitized_mol) -> int:
    """Total radical electrons of a sanitized molecule; 0 when sanitization failed (None).

    RDKit only assigns radicals during sanitization, so an unsanitized molecule always reports 0.
    """
    if sanitized_mol is None:
        return 0
    return sum(atom.GetNumRadicalElectrons() for atom in sanitized_mol.GetAtoms())


def assess_smiles(smiles: str, charge: int) -> Dict[str, Any]:
    """Quality fields for one SMILES, keyed by their metadata column names.

    Args:
        smiles: Atom-mapped SMILES with explicit hydrogens.
        charge: Total molecular charge of the source record.

    smiles_sanitize_error is "" when sanitization succeeds (a string rather than None so every
    shard infers the same column type). smiles_radical is only meaningful when smiles_sanitizes
    is True. An unparsable SMILES gives smiles_sanitize_error="ParseError", zero counts and
    smiles_charge_matches=False.
    """
    mol = parse_mapped_smiles(smiles)
    if mol is None:
        return {
            "smiles_n_fragments": 0,
            "smiles_n_orphaned_h": 0,
            "smiles_sanitizes": False,
            "smiles_sanitize_error": "ParseError",
            "smiles_radical": False,
            "smiles_formal_charge": 0,
            "smiles_charge_matches": False,
        }

    sanitized_mol, sanitize_error = try_sanitize(mol)
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    return {
        "smiles_n_fragments": len(Chem.GetMolFrags(mol)),
        "smiles_n_orphaned_h": count_orphaned_h(mol),
        "smiles_sanitizes": sanitized_mol is not None,
        "smiles_sanitize_error": sanitize_error,
        "smiles_radical": count_radical_electrons(sanitized_mol) > 0,
        "smiles_formal_charge": formal_charge,
        "smiles_charge_matches": formal_charge == charge,
    }
