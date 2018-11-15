
"""
    helper module creating a number of useful functions for manipulating hexidecimal byte representations of
    floats.
    GNS.cri
    15/11/18
"""
import struct
import binascii


def float2bytearray(afloat):
    """convert a float into a mutable bytearray representing it"""
    return bytearray(binascii.hexlify(struct.pack('f', afloat)))

def bytearray2float(bytearray):
    """convert a bytearray into a float"""
    return struct.unpack('f', binascii.unhexlify(bytes(bytearray)))[0]

