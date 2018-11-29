"""some helpful functions for sorting data in MOO applications"""
import itertools as itr

def quickselect(data, k):
    """implements a quickselect algorithim for use in SPEA-2"""
    return __quickselect_helper(data, 0, len(data) - 1, k)


def __quickselect_helper(data, left, right, k):
    if left == right:
        result = data[left]
    else:
        pivot_index = (left + right) // 2
        pivot_index = __partition(data, left, right, pivot_index)
        if k == pivot_index:
            result = data[k]
        elif k < pivot_index:
            result = __quickselect_helper(data, left, pivot_index - 1, k)
        else:
            result = __quickselect_helper(data, pivot_index + 1, right, k)
    return result


def __partition(data, left, right, pivot_index):
    result = None
    pivot = data[pivot_index]
    data[pivot_index] = data[left]
    i = left + 1
    j = right
    not_finished = True
    while not_finished:
        while i <= right and data[i] < pivot:
            i += 1
        while j >= left + 1 and pivot < data[j]:
            j -= 1
        if i >= j:
            not_finished = False
            data[left], data[j] = data[j], pivot
            result = j
        else:
            data[i], data[j] = data[j], data[i]
    return result


def main():
    alist = [1, 2, 3, 4, 5]
    assert quickselect(alist, 0) == 1
    assert quickselect(alist, 1) == 2
    assert quickselect(alist, 2) == 3
    assert quickselect(alist, 3) == 4
    blist = [5, 4, 3, 2, 1]
    assert quickselect(blist, 0) == 1
    assert quickselect(blist, 1) == 2
    assert quickselect(blist, 2) == 3
    assert quickselect(blist, 3) == 4
    clist = [1, 2, 3, 0, 4, 5]
    assert quickselect(clist, 0) == 0
    assert quickselect(clist, 1) == 1
    assert quickselect(clist, 2) == 2
    assert quickselect(clist, 3) == 3
    assert quickselect(clist, 4) == 4
    assert quickselect(clist, 5) == 5
    alist = [1, 3, 4, 2, 3, 4, 9]
    for l in itr.permutations(alist, len(alist)):
        assert quickselect(list(l), 1) == 2


main()