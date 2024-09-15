import numpy as np
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
def mantissaDecimal(mantissa, exponent):
    """
    For a binary mantissa and exponent value in a list form, ie, [1, 0, 1, 0, 0, 0, ...] return the 
    value given if it were an IEEE 754 mantissa and an exponent
    
    Inputs:
    ------
    mantissa : list[int]
    exponent : list[int]
        
    Returns:
    -------
    integer
    
    """
    sum = np.float64(1)
    for i in range(len(mantissa)):
        f = np.flip(mantissa)
        
        sum += f[22-i]*2**(-i-1)
    return sum

#initial value we want to convert to ieee representation
init = 100.98763
#ieee representation of the number
representation = get_representation(100.98763)
print("ieee representation:" )
print(representation)

#Finding exponent and value of representation
exponent = int(int(int(''.join(str(item) for item in representation["exponent"]), 2)))-127
value = mantissaDecimal(representation["mantissa"], exponent)

#converting representation to a decimal value and finding the percent differencce from the initial
bintoDec = value*2**exponent
print("conversion from binary representation to decimal:" + str(bintoDec))
percentdifference = 100*np.abs(bintoDec-init)/init

print("percent difference: " +str(percentdifference))
