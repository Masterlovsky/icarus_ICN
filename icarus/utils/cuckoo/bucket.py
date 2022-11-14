"""
Cuckoo filter internal bucket.
"""

import random


class Bucket(object):
    """
    Bucket class for storing fingerprints.
    """
    # https://docs.python.org/3/reference/datamodel.html#object.__slots__
    __slots__ = ('size', 'bucket')

    def __init__(self, size=4):
        """
        Initialize a dynamic or static bucket to keep a set of Cuckoo
        fingerprints.

        size: The maximum number of fingerprints the bucket can store.
              Default size is 4, which closely approaches the best size for FPP
                between 0.00001 and 0.002 (see Fan et al.).
              If your targeted FPP is greater than 0.002, a bucket size of 2 is
                more space efficient.
        """
        self.size = size

        # The bucket is implemented as an array because it's possible to have
        # multiple items with the same fingerprints
        self.bucket = []
        # TODO: investigate a better way to compress the bucket's fingerprints.
        # It will be very helpful when long fingerprints are need for low error
        # rate.

    def insert(self, fingerprint):
        """
        Insert a fingerprint into the bucket, the fingerprint basically is just
        a bit vector.  The longer the bit vector, the lower the collision rate.

        The insertion of duplicate entries is allowed.
        """
        if self.contains(fingerprint):
            return True
        if not self.is_full():
            self.bucket.append(fingerprint)
            # When the bucket is not full, just append the fingerprint there
            return True

        # In static mode, the size of the bucket is fixed.  It means that the
        # filter is reaching its capacity here.
        return False

    def contains(self, fingerprint):
        """
        Check if this bucket contains the provided fingerprint.
        """
        return fingerprint in self.bucket

    def find_and_replace(self, look_for, replace_with):
        """
        Find an exact fingerprint the specified bucket and replace it with
        another fingerprint.  Return False if there is no such fingerprint.
        """
        try:
            self.bucket[self.bucket.index(look_for)] = replace_with
            return True

        except ValueError:
            # No such fingerprint in the bucket
            return False

    def delete(self, fingerprint):
        """
        Delete a fingerprint from the bucket.

        Returns True if the fingerprint was present in the bucket. This is
        useful for keeping track of how many items are present in the filter.
        """
        try:
            del self.bucket[self.bucket.index(fingerprint)]
            return True

        except ValueError:
            # No such fingerprint in the bucket
            return False

    def swap(self, fingerprint):
        """
        Swap a fingerprint with a randomly chosen fingerprint from the bucket.

        The given fingerprint is stored in the bucket.
        The swapped fingerprint is returned.
        """
        # There is tricky bug in swap function when an item is added several
        # times. In such case, there is a chance that a fingerprint is swapped
        # with itself thus trying to move fingerprints around won't work.
        #
        # Assuming that the bucket size is 4, the maximum number of times an
        # item can be added is 4 * 2 = 8.
        #
        # TODO: Investigate if there is a better solution for this cause this
        # is a form of local limit of Cuckoo filter.
        rindex = random.choice([i for i in range(len(self.bucket))
                                if fingerprint != self.bucket[i]])

        # Swap the two fingerprints
        fingerprint, self.bucket[rindex] = self.bucket[rindex], fingerprint
        # and return the one from the bucket
        return fingerprint

    def is_full(self):
        """
        Signify that the bucket is full, a fingerprint will need to be swapped
        out.
        """
        return len(self.bucket) >= self.size

    def __contains__(self, fingerprint):
        return self.contains(fingerprint)

    def __repr__(self):
        return '<Bucket: {0}>'.format(self.bucket)

    def __sizeof__(self):
        return super().__sizeof__() + sum(f.__sizeof__() for f in self.bucket)


class SuperBucket(Bucket):
    """
    Super Bucket class for storing both fingerprint and bit string for set identifications
    """
    __slots__ = ('size', 'bucket', 'bit_string')

    def __init__(self, size=4):
        """
        Initialize a dynamic or static bucket to keep a set of Cuckoo
        fingerprints.

        size: The maximum number of fingerprints the bucket can store.
              Default size is 4, which closely approaches the best size for FPP
                between 0.00001 and 0.002 (see Fan et al.).
              If your targeted FPP is greater than 0.002, a bucket size of 2 is
                more space efficient.
        """
        super().__init__(size)

        # The bucket is implemented as an array because it's possible to have
        # multiple items with the same fingerprints
        self.bucket = []

    def insert(self, fingerprint):
        return super().insert(fingerprint)

    def contains(self, fingerprint):
        return super().contains(fingerprint)

    def find_and_replace(self, look_for, replace_with):
        return super().find_and_replace(look_for, replace_with)

    def delete(self, fingerprint):
        return super().delete(fingerprint)

    def swap(self, fingerprint):
        return super().swap(fingerprint)

    def is_full(self):
        return super().is_full()

    def __contains__(self, fingerprint):
        return super().__contains__(fingerprint)

    def __repr__(self):
        return super().__repr__()

    def __sizeof__(self):
        return super().__sizeof__()

