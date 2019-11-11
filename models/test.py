import Levenshtein as Lev

def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 

ref = '가나다라마바사'
hyp = '가나다바마바사'

print(char_distance(ref,hyp))