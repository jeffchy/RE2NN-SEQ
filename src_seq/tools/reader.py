def MIT_BIO_READER(f):
    """
    :param f: file object
    :return:
    """

    querys = []
    tags = []
    q = []
    t = []
    wc = 0
    sc = 0
    for line in f.readlines():
        line = line.strip()
        if line:
            _t, _q = line.split()
            q.append(_q.lower())
            t.append(_t.lower())
            wc += 1
        else:
            querys.append(q)
            tags.append(t)
            q = []
            t = []
            sc += 1

    print('total words: {}, total sentences: {}'.format(wc, sc))
    return querys, tags


def SNIPS_BIO_READER(f):
    """
    :param f: file object
    :return:
    """

    querys = []
    tags = []
    q = []
    t = []
    wc = 0
    sc = 0
    for line in f.readlines():
        line = line.strip()
        if line:
            if len(line.split()) > 1:
                _q, _t = line.split()
                q.append(_q.lower())
                t.append(_t.lower())
                wc += 1
        else:
            querys.append(q)
            tags.append(t)
            q = []
            t = []
            sc += 1

    print('total words: {}, total sentences: {}'.format(wc, sc))
    return querys, tags


def CONLL03_BIO_READER(f):
    """
    :param f: file object
    :return:
    """

    querys = []
    tags = []
    q = []
    t = []
    wc = 0
    sc = 0
    for line in f.readlines():
        line = line.strip()
        if line:
            split_line = line.split()
            _t, _q = split_line[3], split_line[0]
            q.append(_q.lower())
            t.append(_t.lower())
            wc += 1
        else:
            querys.append(q)
            tags.append(t)
            q = []
            t = []
            sc += 1

    print('total words: {}, total sentences: {}'.format(wc, sc))
    return querys, tags
