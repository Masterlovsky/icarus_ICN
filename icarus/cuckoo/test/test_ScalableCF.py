from icarus.cuckoo.filter import *
import sys

if __name__ == '__main__':
    cuckoo = ScalableCuckooFilter(10000, 0.02, 4, class_type=MarkedCuckooFilter)
    succ_count = 0
    check_count = 0
    for i in range(10000):
        try:
            cuckoo.insert(str(i), mask="0"*15+"1"+"0"*32)
            succ_count += 1
        except Exception as e:
            print(e)
    print(succ_count)

    size = 100000
    for i in range(20000, 20000 + size):
        if cuckoo.contains(str(i)):
            check_count += 1
    print("False positive number {}/{}".format(check_count, size))
    print("Filter numbers: ", cuckoo.get_filter_nums())
    print("Fingerprint size : ", cuckoo.get_fingerprint_size())
    print("size of cuckoo : ", sys.getsizeof(cuckoo))
    # print("=============")
    # bcf = BCuckooFilter(10000, 0.02, 4)
    # cf = CuckooFilter(10000, 0.02, 4)
    # print("size of bcf: {}, fingerprint size: {}".format(sys.getsizeof(bcf), bcf.get_fingerprint_size()))
    # print("size of cf: {}, fingerprint size: {}".format(sys.getsizeof(cf), cf.get_fingerprint_size()))

    print("=============")
    # cuckoo = MarkedCuckooFilter(10000, 0.02, 4, bit_tag_len=16)
    # print("cuckoo size: {}B".format(sys.getsizeof(cuckoo)))
    # print(cuckoo)
    # cuckoo.insert_m("1", cuckoo.encode_mask('bit', 3))
    # print("check result: ", cuckoo.contains("1"))
    # print("check result 1: ", cuckoo.decode_mask(cuckoo.get("1")))
    # cuckoo.insert_m("2", cuckoo.encode_mask('int', 65530))
    # print("check result 2: ", cuckoo.decode_mask(cuckoo.get("2")))
    # for i in range(3, 100):
    #     cuckoo.insert_m(str(i), cuckoo.encode_mask('int', i))
    #     print("check result {}: {}".format(i, cuckoo.decode_mask(cuckoo.get(str(i)))))