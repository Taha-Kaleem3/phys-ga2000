import numpy as np
print("the binary value of the representation is ")
#problem 3.40282*10^38 for problem 2
def get_bits(number):
    """For a NumPy quantity, return bit representation, this is done using the IEEE 754 standard 
    which is described in my solution to the problem set.
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

    
def get_representation(value):
    """For a NumPy quantity, return IEEE 754 standard for a given value using the get_bits function.
    
    Inputs:
    ------
    value : NumPy value (32 but float)
        value to convert into list of bits
        
    Returns:
    -------
    temlate : list
       list of 0 and 1 values, highest to lowest significance
    """
    bitlist=get_bits(np.float32(value))
    sign = bitlist[0]
    exponent = bitlist[1:9]
    mantissa = bitlist[9:32]
    template = {"value": value,
       "sign" : sign, 
       "exponent": exponent, 
       "mantissa": mantissa}

    return template

print(get_representation(100.98763))
for value in [100.98763]:
    bitlist=get_bits(value)
    sign = bitlist[0]
    exponent = bitlist[1:9]
    mantissa = bitlist[9:32]
    template = {"value": value,
       "sign" : sign, 
       "exponent": exponent, 
       "mantissa": mantissa}
