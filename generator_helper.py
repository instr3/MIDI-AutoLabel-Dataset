
def end_generator(generator):
    try:
        next(generator)
        raise ValueError('Generator did not end')
    except StopIteration as e:
        return e.value

def next_tag(gen, expected_tag):
    data = next(gen)
    assert data[0] == expected_tag, f"Expected tag {expected_tag}, got {data[0]}"
    return data

def injection(tag, *args):
    container = [tag, *args]
    yield container
    return container