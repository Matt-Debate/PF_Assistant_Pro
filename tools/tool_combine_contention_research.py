"""
tool_combine_contention_research.py

Combines the polished contention text and the formatted evidence cards text
into a single output string for final display to the user.
"""

def combine_contention_and_evidence(polished_contention: str, formatted_evidence: str) -> str:
    """
    Combine the polished contention string and the formatted evidence cards string
    into a single output for final display.

    Args:
        polished_contention (str): Polished and revised contention text.
        formatted_evidence (str): Plain text formatted evidence cards.

    Returns:
        str: Combined final output with contention followed by evidence cards.
    """
    combined_output = polished_contention.strip() + "\n\n" + formatted_evidence.strip()
    return combined_output
