# token_counter.py
total_tokens_used = 0

def add_tokens(count):
    global total_tokens_used
    total_tokens_used += count

def get_total_tokens():
    return total_tokens_used
