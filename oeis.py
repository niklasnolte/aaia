"""Access to the Online Encylopedia of Integer Sequences (OEIS)

The OEIS is accessible via a web interface at http://oeis.org/

This module uses requests to access the OEIS, and then parses a minimal
subset of the information contained there.

If you want to search for the 5 best matches for the sequence [1,2,3,4,5],
you could do something like:

>>> import oeis
>>> sequences = oeis.query([1, 2, 3, 4, 5], 5)
>>> for seq in sequences:
>>>    print seq.id
>>>    print seq.name
>>>    print seq.formula
>>>    print seq.sequence
>>>    print seq.comments

This code is copyright 2013 Andrew Walker
See the end of the source file for the license of use.
"""
import requests
__version__ = '0.1'
__all__ = ['query']


class IntegerSequence(object):
    """Integer sequence API around the OEIS internal format

    References
    ----------
    [1] http://oeis.org/eishelp1.html
    """
    def __init__(self, blk):
        """Parse the OEIS text data"""
        self.id = None
        self.name = None
        self.formula = None
        self.sequence = []
        self.comments = ''

        for line in blk:
            # TODO - adopt a more robust approach to this split
            data_type = line[1]
            sequence_name = line[3:10]
            data = line[10:].strip()

            if data_type == 'I':
                self.id = sequence_name
            elif data_type == 'S':
                if data[-1] == ',':
                    data = data[:-1]
                self.sequence = [int(num) for num in data.split(',')]
            elif data_type == 'N':
                self.name = data
            elif data_type == 'C':
                self.comments += (data + '\n')
            elif data_type == 'F':
                self.formula = data


class OEISError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


def raw_query(sequence, n=1):
    """Execute a raw query to the OEIS database
    """
    payload = {}
    payload['q'] = ','.join(str(s) for s in sequence)
    payload['n'] = str(n)
    payload['fmt'] = 'text'
    response = requests.get('http://oeis.org/search', params=payload)
    if response.status_code != 200:
        raise OEISError('Invalid HTTP response from the OEIS')
    return response.content.decode()


def split_blocks(content):
    """Split the response text into blocks related to each sequence
    """
    blocks = content.split("\n\n")
    return [block for block in blocks if _valid_block(block)]


def _valid_block(block):
    """Identify Valid blocks of sequence text

    A valid block must be non-empty, and start with
    an appropriate marker (%)
    """
    return len(block) > 0 and block.startswith('%')


def query(sequence, n=1):
    """Search the OEIS for upto `n` of the best matches for a sequence
    """
    content = raw_query(sequence, n)
    blocks = split_blocks(content)
    return [IntegerSequence(block.split('\n')) for block in blocks]

class OEIS:
    def __init__(self):
        self.all_sequences = []
        # read
