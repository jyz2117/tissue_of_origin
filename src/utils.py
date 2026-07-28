def time_convert(time: float) -> str:
    """
    Converts a float time (in seconds) into a human-readable string.

    Args:
        time (float): Time in seconds.

    Returns:
        str: Formatted time string (seconds, minutes, or hours).
    """
    if time < 60:
        return f"{time:.2f} seconds"
    elif time < 3600:
        return f"{time/60:.2f} minutes"
    else:
        return f"{time/3600:.2f} hours"