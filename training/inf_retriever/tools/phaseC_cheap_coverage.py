"""Share of each full-corpus dev retrieval (top-100 / top-10) that lies inside the reduced
cheap-eval corpus: low coverage means cheap eval is missing the distractors the model really retrieves."""
import json,sys,glob,csv
csv.field_size_limit(10**9)
ids=set()
with open("data/phaseA/corpora/qampari_dev500.tsv") as f:
    next(f)
    for l in f: ids.add(l.split("\t",1)[0])
for d in sorted(glob.glob("results/phaseC/full/qampari/*/dev")):
    fs=[x for x in glob.glob(d+"/*.jsonl")]
    if not fs: continue
    tot=inn=0; top10=top10in=0
    for l in open(fs[0]):
        c=json.loads(l)["ctxs"][:100]
        tot+=len(c); inn+=sum(x["id"] in ids for x in c)
        top10+=min(10,len(c)); top10in+=sum(x["id"] in ids for x in c[:10])
    print("%-40s top-100 in reduced corpus: %.1f%%   top-10: %.1f%%"%(d.split("/")[-2],100*inn/tot,100*top10in/top10))
