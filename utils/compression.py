import zlib
import pickle
import base64

def compress(raw_data):
    serialized_data = pickle.dumps(raw_data)
    compressed_data = zlib.compress(serialized_data, level=9)
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    return encoded_data

def decompress(compressed_data):
    decoded_data = base64.b64decode(compressed_data)
    decompressed_data = zlib.decompress(decoded_data)
    raw_data = pickle.loads(decompressed_data)
    return raw_data