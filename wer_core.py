import random
import re

SENTENCE_MAP = {
    'IEO': "it's eleven o'clock",
    'TIE': 'that is exactly what happened',
    'IOM': "i'm on my way to the meeting",
    'IWW': 'i wonder what this is about',
    'TAI': 'the airplane is almost full',
    'MTI': 'maybe tomorrow it will be cold',
    'IWL': 'i would like a new alarm clock',
    'ITH': "i think i have a doctor's appointment",
    'DFA': "don't forget a jacket",
    'ITS': "i think i've seen this before",
    'TSI': 'the surface is slick',
    'WSI': "we'll stop in a couple of minutes",
}


def norm(t):
    t = t.lower().replace("'", "")
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return ' '.join(t.split())


def edist(r, h):
    r, h = r.split(), h.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)]


def bootstrap_ci(items, n_boot=1000, seed=42):
    rng = random.Random(seed)
    n = len(items)
    vals = []
    for _ in range(n_boot):
        e = w = 0
        for _ in range(n):
            ei, wi = items[rng.randrange(n)]
            e += ei
            w += wi
        vals.append(100 * e / max(1, w))
    vals.sort()
    return round(vals[int(0.025 * (n_boot - 1))], 2), round(vals[int(0.975 * (n_boot - 1))], 2)


def gt_recon_wer_summary(results, stem_key='file', hyp_key='hypothesis'):
    items = []
    per = []
    for r in results:
        gt = SENTENCE_MAP.get(r[stem_key].split('_')[1])
        if gt is None:
            continue
        e = edist(norm(gt), norm(r[hyp_key]))
        n = len(norm(gt).split())
        r['reference'] = gt
        r['wer'] = e / max(1, n)
        items.append((e, n))
        per.append(e / max(1, n))
    te = sum(e for e, _ in items)
    tw = sum(n for _, n in items)
    lo, hi = bootstrap_ci(items)
    return {
        'corpus_wer_pct': round(100 * te / max(1, tw), 2),
        'per_sample_wer_pct': round(100 * sum(per) / max(1, len(per)), 2),
        'n': len(items),
        'ci_low': lo,
        'ci_high': hi,
    }
