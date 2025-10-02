def partition(iterable, ratio, split_gap=0):
    """
    Split the iterable into two proportionally to ratio. Optionally ensure
    an aboslute gap ahead of the split point to prevent window overlapping.
    
    Args:
        iterable (Iterable): The iterable to split
        ratio (float): The split ratio
        split_gap (int): The index gap between the two splits
    
    Returns:
        (Iterable, Iterable): The split segments
    """
    return (
        iterable[:int(len(iterable)*ratio) - 1],
        iterable[int(len(iterable)*ratio) - 1 + split_gap:]
    )

def max_divisor(val : int, max : int):
    """
    Return the largest factor of val less than or equal to max.
    """
    ...