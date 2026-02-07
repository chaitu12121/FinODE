"""
Crypto Vault

AES-GCM encryption for sensitive data.
"""

import os
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2


class CryptoVault:
    """
    Encryption/decryption vault using AES-256-GCM.
    """
    
    def __init__(self, master_key: bytes = None):
        """
        Initialize crypto vault.
        
        Args:
            master_key: 32-byte key (if None, generates random)
        """
        if master_key is None:
            self.master_key = AESGCM.generate_key(bit_length=256)
        else:
            if len(master_key) != 32:
                raise ValueError("Master key must be 32 bytes")
            self.master_key = master_key
    
    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt plaintext using AES-256-GCM.
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            (ciphertext, nonce, tag)
        """
        nonce = os.urandom(12)  # 96-bit nonce
        cipher = AESGCM(self.master_key)
        ciphertext = cipher.encrypt(nonce, plaintext, None)
        return ciphertext, nonce, ciphertext[-16:]  # Last 16 bytes are tag
    
    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
        """
        Decrypt ciphertext using AES-256-GCM.
        
        Args:
            ciphertext: Data to decrypt
            nonce: Nonce used in encryption
            tag: Authentication tag
        
        Returns:
            Plaintext
        """
        cipher = AESGCM(self.master_key)
        plaintext = cipher.decrypt(nonce, ciphertext, None)
        return plaintext
    
    def derive_key(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Derive 256-bit key from password using PBKDF2.
        
        Args:
            password: Password string
            salt: Random salt (generated if None)
        
        Returns:
            (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = kdf.derive(password.encode())
        return key, salt
