"""
DNA Cryptography Module
Handles all DNA encoding, chaotic maps, encryption, and decryption.
"""

import numpy as np
import hashlib
import json

ENCODING = {
    1: {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},
    2: {'00': 'A', '01': 'G', '10': 'C', '11': 'T'},
    3: {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},
    4: {'00': 'C', '01': 'T', '10': 'A', '11': 'G'},
    5: {'00': 'G', '01': 'A', '10': 'T', '11': 'C'},
    6: {'00': 'G', '01': 'T', '10': 'A', '11': 'C'},
    7: {'00': 'T', '01': 'C', '10': 'G', '11': 'A'},
    8: {'00': 'T', '01': 'G', '10': 'C', '11': 'A'},
}

DECODING = {
    r: {v: k for k, v in rule.items()}
    for r, rule in ENCODING.items()
}

XOR = {
    ('A','A'):'A', ('A','C'):'C', ('A','G'):'G', ('A','T'):'T',
    ('C','A'):'C', ('C','C'):'A', ('C','G'):'T', ('C','T'):'G',
    ('G','A'):'G', ('G','C'):'T', ('G','G'):'A', ('G','T'):'C',
    ('T','A'):'T', ('T','C'):'G', ('T','G'):'C', ('T','T'):'A',
}

ADD = {
    ('A','A'):'A', ('A','C'):'C', ('A','G'):'G', ('A','T'):'T',
    ('C','A'):'C', ('C','C'):'G', ('C','G'):'T', ('C','T'):'A',
    ('G','A'):'G', ('G','C'):'T', ('G','G'):'A', ('G','T'):'C',
    ('T','A'):'T', ('T','C'):'A', ('T','G'):'C', ('T','T'):'G',
}

SUB = {
    ('A','A'):'A', ('A','C'):'T', ('A','G'):'G', ('A','T'):'C',
    ('C','A'):'C', ('C','C'):'A', ('C','G'):'T', ('C','T'):'G',
    ('G','A'):'G', ('G','C'):'C', ('G','G'):'A', ('G','T'):'T',
    ('T','A'):'T', ('T','C'):'G', ('T','G'):'C', ('T','T'):'A',
}

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary) - 7, 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

def binary_to_dna(binary, rule=1):
    if len(binary) % 2 != 0:
        binary += '0'
    return ''.join(ENCODING[rule][binary[i:i+2]] for i in range(0, len(binary), 2))

def dna_to_binary(dna, rule=1):
    return ''.join(DECODING[rule][b] for b in dna)

def logistic_map(x0, r, n):
    seq = []
    x = x0
    for _ in range(n):
        x = r * x * (1 - x)
        seq.append(x)
    return seq

def chaotic_to_dna(seq):
    bases = ['A', 'C', 'G', 'T']
    return ''.join(bases[int(v * 4) % 4] for v in seq)

def make_key(password, length):
    h = hashlib.sha256(password.encode()).hexdigest()
    b = bin(int(h, 16))[2:].zfill(256)
    while len(b) < length * 2:
        b += b
    return binary_to_dna(b[:length * 2], rule=1)

def dna_xor(s1, s2):
    return ''.join(XOR[(a, b)] for a, b in zip(s1, s2))

def dna_add(s1, s2):
    return ''.join(ADD[(a, b)] for a, b in zip(s1, s2))

def dna_sub(s1, s2):
    return ''.join(SUB[(a, b)] for a, b in zip(s1, s2))

def encrypt(plaintext, password, rule=1, x0=0.5, r=3.99):
    binary = text_to_binary(plaintext)
    dna = binary_to_dna(binary, rule)
    pk = make_key(password, len(dna))
    ck = chaotic_to_dna(logistic_map(x0, r, len(dna)))
    combined = dna_xor(pk, ck)
    enc = dna_xor(dna, combined)
    final = dna_add(enc, ck)
    return {'cipher': final, 'rule': rule, 'x0': x0, 'r': r}

def decrypt(cipher_dict, password):
    final = cipher_dict['cipher']
    rule = cipher_dict['rule']
    x0 = cipher_dict['x0']
    r = cipher_dict['r']
    ck = chaotic_to_dna(logistic_map(x0, r, len(final)))
    enc = dna_sub(final, ck)
    pk = make_key(password, len(enc))
    combined = dna_xor(pk, ck)
    dna = dna_xor(enc, combined)
    binary = dna_to_binary(dna, rule)
    return binary_to_text(binary)

def encrypt_weights(weights_dict, password, rule=1, x0=0.5, r=3.99):
    """
    DNA Encryption for neural network weights.
    Uses DNA nucleotide encoding, SHA-256 key schedule, and chaotic logistic maps.
    Vectorized for high-performance federated learning over deep CNN models.
    """
    encrypted = {}
    for name, w in weights_dict.items():
        shape = list(w.shape)
        dtype_str = str(w.dtype)
        # Convert float weights to raw bytes (lossless, compact)
        raw_bytes = w.astype(np.float32).tobytes()
        
        # Each byte (8 bits) represents 4 DNA bases (2 bits each)
        byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        n_bases = len(byte_arr) * 4
        
        b0 = (byte_arr >> 6) & 3
        b1 = (byte_arr >> 4) & 3
        b2 = (byte_arr >> 2) & 3
        b3 = byte_arr & 3
        dna_int = np.column_stack((b0, b1, b2, b3)).reshape(-1)
        
        # 1. Chaotic Keystream from Logistic Map
        chaotic_vals = np.array(logistic_map(x0, r, n_bases), dtype=np.float64)
        ck = ((chaotic_vals * 4).astype(np.int64)) % 4
        
        # 2. Key Derivation from Password using SHA-256
        h = hashlib.sha256(password.encode()).hexdigest()
        pk_seed = (int(h[:8], 16) % 900000 + 100000) / 1000000.0  # Seed in (0, 1)
        pk_vals = np.array(logistic_map(pk_seed, r, n_bases), dtype=np.float64)
        pk = ((pk_vals * 4).astype(np.int64)) % 4
        
        # 3. DNA Algebraic Operations
        combined = pk ^ ck               # DNA XOR
        enc = dna_int ^ combined         # DNA XOR
        final = (enc + ck) % 4           # DNA Addition (mod 4)
        
        # Re-pack 4 bases into uint8
        final_2d = final.reshape(-1, 4)
        cipher_bytes = (
            (final_2d[:, 0] << 6) |
            (final_2d[:, 1] << 4) |
            (final_2d[:, 2] << 2) |
            final_2d[:, 3]
        ).astype(np.uint8).tobytes()
        
        encrypted[name] = {
            'cipher_hex': cipher_bytes.hex(),
            'shape': shape,
            'dtype': dtype_str,
            'rule': rule,
            'x0': x0,
            'r': r
        }
    return encrypted

def decrypt_weights(encrypted_dict, password):
    """
    DNA Decryption for neural network weights.
    Reverses chaotic DNA addition, XOR, and nucleotide decoding.
    """
    weights = {}
    for name, enc in encrypted_dict.items():
        if 'cipher_hex' in enc:
            cipher_bytes = bytes.fromhex(enc['cipher_hex'])
            byte_arr = np.frombuffer(cipher_bytes, dtype=np.uint8)
            n_bases = len(byte_arr) * 4
            
            b0 = (byte_arr >> 6) & 3
            b1 = (byte_arr >> 4) & 3
            b2 = (byte_arr >> 2) & 3
            b3 = byte_arr & 3
            final = np.column_stack((b0, b1, b2, b3)).reshape(-1)
            
            x0 = enc['x0']
            r = enc['r']
            chaotic_vals = np.array(logistic_map(x0, r, n_bases), dtype=np.float64)
            ck = ((chaotic_vals * 4).astype(np.int64)) % 4
            
            h = hashlib.sha256(password.encode()).hexdigest()
            pk_seed = (int(h[:8], 16) % 900000 + 100000) / 1000000.0
            pk_vals = np.array(logistic_map(pk_seed, r, n_bases), dtype=np.float64)
            pk = ((pk_vals * 4).astype(np.int64)) % 4
            
            # Reverse: enc = (final - ck) % 4
            enc_data = (final - ck) % 4
            combined = pk ^ ck
            dna_int = enc_data ^ combined
            
            dna_2d = dna_int.reshape(-1, 4)
            orig_bytes = (
                (dna_2d[:, 0] << 6) |
                (dna_2d[:, 1] << 4) |
                (dna_2d[:, 2] << 2) |
                dna_2d[:, 3]
            ).astype(np.uint8).tobytes()
            
            shape = enc['shape']
            dtype_str = enc.get('dtype', 'float32')
            weights[name] = np.frombuffer(orig_bytes, dtype=np.float32).reshape(shape).astype(dtype_str)
        else:
            # Fallback for legacy JSON string format
            text = decrypt(enc, password)
            flat = json.loads(text)
            shape = enc['shape']
            dtype = enc.get('dtype', 'float32')
            weights[name] = np.array(flat, dtype=dtype).reshape(shape)
    return weights
