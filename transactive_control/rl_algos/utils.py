
#Helper function
def string2bool(input_str: str):
    """
    Purpose: Convert strings to boolean. Specifically 'T' -> True, 'F' -> False
    """
    input_str = input_str.upper()
    if(input_str == 'T'):
        return True
    elif(input_str == 'F'):
        return False
    else:
        raise NotImplementedError ("Unknown boolean conversion for string: {}".format(input_str))


