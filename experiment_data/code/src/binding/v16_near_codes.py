"""Fixed one-digit foils with no leading zero or in-record collision."""
import hashlib
import re

def near_code(gold,record_values,key,position):
    if position not in (0,3) or not re.fullmatch('[1-9][0-9]{3}',gold):
        raise ValueError('Four-digit gold; first/last position only')
    values=[]
    for digit in ('123456789' if position==0 else '0123456789'):
        candidate=gold[:position]+digit+gold[position+1:]
        if candidate!=gold and candidate not in record_values:values.append(candidate)
    if not values:raise ValueError('No available out-of-record one-digit foil')
    index=int(hashlib.sha256((key+':'+str(position)).encode()).hexdigest()[:16],16)%len(values)
    return values[index]
