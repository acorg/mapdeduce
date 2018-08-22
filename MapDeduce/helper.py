"""Helper functions."""

from data import amino_acids

def is_not_amino_acid(a):
    """Test if a is not an amino acid.
    
    Args:
        a (str): String to test.

    Returns:
        bool. True if a is not an amino acid.
    """
    try:
        a = a.upper()
    except AttributeError:
        return True

    if a not in amino_acids:
        return True

    return False
